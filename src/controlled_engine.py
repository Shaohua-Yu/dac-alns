"""Controlled, auditable reruns of the snapshotted ALNS / AC / DAC solvers.

All methods use the DAC problem environment and identical nearest-neighbour
initialization. Original source is never edited. Imports and process warmup are
outside trial time; instance parsing, initialization, network construction and
checkpoint loading are inside it. CPU only, one torch thread per worker.

Public API: run_trial(config, agent=None), create_agent(...), save_checkpoint(...),
load_checkpoint(...), warmup_backend(problem). A supplied agent is retained with
its optimizer state between training instances. Never use this option in tests.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "original_source"
sys.path.insert(0, str(SOURCE))
ENGINE_VERSION = "controlled-20261008-v1"
METHODS = {"ALNS", "AC", "DAC-Partial", "DAC-Pre", "DAC-Credit", "Frozen", "Adapt"}
_MODULES = {}


def _module(problem, single=False):
    if problem not in {"CVRP", "VRPTW"}:
        raise ValueError(f"Unknown problem: {problem}")
    name = f"AC_{problem}" if single else f"DAC_{problem}_online"
    if name not in _MODULES:
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != SOURCE.resolve():
            raise RuntimeError(f"Expected archived source, got {module.__file__}")
        module.device = torch.device("cpu")
        _MODULES[name] = module
    return _MODULES[name]


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_agent(problem, method="DAC-Partial", seed=0, checkpoint_path=None):
    """Create original architecture; loading weights resets optimizers."""
    module = _module(problem, method == "AC")
    _seed(int(seed))
    cls = module.SingleACAgent if method == "AC" else module.DACAgent
    agent = cls(enc_dim=64, hid=64, gamma_repair=0.99, critic_coef=1.0).cpu()
    agent._controlled_problem = problem
    if checkpoint_path:
        load_checkpoint(agent, checkpoint_path)
    agent.train()
    return agent


def load_checkpoint(agent, path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    weights = checkpoint.get("agent_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    problem = checkpoint.get("problem") if isinstance(checkpoint, dict) else None
    if problem and problem != getattr(agent, "_controlled_problem", problem):
        raise ValueError("Checkpoint problem mismatch")
    agent.load_state_dict(weights, strict=True)
    return checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}


def save_checkpoint(agent, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"engine_version": ENGINE_VERSION,
                "problem": getattr(agent, "_controlled_problem", None),
                "agent_state_dict": agent.state_dict(),
                "metadata": metadata or {}}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def warmup_backend(problem):
    """Compile problem-independent DP signatures before any timed trials."""
    module = _module(problem)
    _module(problem, True)
    seq = np.array([1], dtype=np.int32)
    dem = np.array([1.0], dtype=np.float32)
    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    if problem == "CVRP":
        module._split_dp_core(seq, dem, 10.0, dist, depot_idx=0)
    else:
        # Building a tiny valid model reuses the exact original signature.
        module._split_dp_core(seq, np.array([0.0, 1.0], dtype=np.float32),
                             np.zeros(2, dtype=np.float32), np.full(2, 100.0, dtype=np.float32),
                             np.zeros(2, dtype=np.float32), 10.0, dist, dist,
                             depot_idx=0, depot_start=0.0, depot_end=100.0)
    # PyTorch Adam imports auxiliary modules on first construction.
    create_agent(problem, seed=0)


def _stats(tensor):
    value = tensor.detach().flatten().cpu()
    return {"mean": float(value.mean()), "std": float(value.std(unbiased=False)),
            "min": float(value.min()), "max": float(value.max()),
            "zero_fraction": float((value.abs() < 1e-12).float().mean()),
            "clip_fraction": float((value.abs() >= 1.0).float().mean())}


def _corr(a, b):
    a, b = a.detach().flatten(), b.detach().flatten()
    if a.numel() < 2 or float(a.std(unbiased=False)) < 1e-12 or float(b.std(unbiased=False)) < 1e-12:
        return None
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _update(agent, batch, method, entropy_coef):
    """Original joint/shared-target update, with explicit Pre/Credit switches."""
    full = agent.encode_state_records([r["full"] for r in batch])
    value_full = agent.critic(full)
    rewards = torch.tensor([r["reward"] for r in batch], dtype=torch.float32).view(-1, 1)
    dones = torch.tensor([r["done"] for r in batch], dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        next_full = agent.encode_state_records([r["next"] for r in batch])
        target = rewards + agent.gamma_repair * (1.0 - dones) * agent.critic(next_full)
    td_full = target - value_full
    critic_loss = F.smooth_l1_loss(value_full, target)
    adv_full_raw = td_full.detach()
    diag = {"batch_size": len(batch), "terminal_count": int(dones.sum()),
            "reward": _stats(rewards), "entropy_coefficient": float(entropy_coef)}
    if method == "AC":
        adv_full = agent._normalize_advantage(adv_full_raw).clamp(-5.0, 5.0)
        distribution = agent._policy_distribution(agent.actor_joint(full))
        actions = torch.tensor([r["ad"] * 5 + r["ar"] for r in batch])
        entropy = distribution.entropy().mean()
        actor_loss = -(distribution.log_prob(actions).view(-1, 1) * adv_full).mean() - entropy_coef * entropy
        diag.update(joint_entropy=float(entropy.detach()), destroy_advantage_raw=_stats(adv_full_raw),
                    destroy_advantage=_stats(adv_full))
        components = [(agent.encoder, agent.opt_enc, "encoder"),
                      (agent.critic, agent.opt_critic, "critic"),
                      (agent.actor_joint, agent.opt_joint, "joint")]
    else:
        partial = agent.encode_state_records([r["partial"] for r in batch])
        value_partial = agent.critic(partial)
        td_partial = target - value_partial
        critic_loss = critic_loss + F.smooth_l1_loss(value_partial, target)
        if method == "DAC-Credit":
            adv_full_raw = (value_partial - value_full).detach()
        adv_partial_raw = td_partial.detach()
        adv_full = agent._normalize_advantage(adv_full_raw).clamp(-5.0, 5.0)
        adv_partial = agent._normalize_advantage(adv_partial_raw).clamp(-5.0, 5.0)
        # Critic sees real partial state; Pre's actor sees the frozen full graph.
        repair_input = agent.encode_state_records([r["repair_actor"] for r in batch]) if method == "DAC-Pre" else partial
        dd = agent._policy_distribution(agent.actor_destroy(full))
        rd = agent._policy_distribution(agent.actor_repair(repair_input))
        ads = torch.tensor([r["ad"] for r in batch])
        ars = torch.tensor([r["ar"] for r in batch])
        de, re = dd.entropy().mean(), rd.entropy().mean()
        dloss = -(dd.log_prob(ads).view(-1, 1) * adv_full).mean() - entropy_coef * de
        rloss = -(rd.log_prob(ars).view(-1, 1) * adv_partial).mean() - entropy_coef * re
        actor_loss = dloss + rloss
        diag.update(destroy_entropy=float(de.detach()), repair_entropy=float(re.detach()),
                    destroy_advantage_raw=_stats(adv_full_raw), repair_advantage_raw=_stats(adv_partial_raw),
                    destroy_advantage=_stats(adv_full), repair_advantage=_stats(adv_partial),
                    advantage_raw_correlation=_corr(adv_full_raw, adv_partial_raw),
                    value_correlation=_corr(value_full, value_partial))
        components = [(agent.encoder, agent.opt_enc, "encoder"),
                      (agent.critic, agent.opt_critic, "critic"),
                      (agent.actor_destroy, agent.opt_ad, "destroy"),
                      (agent.actor_repair, agent.opt_ar, "repair")]
    total = agent.critic_coef * critic_loss + actor_loss
    for _, optimizer, _ in components:
        optimizer.zero_grad()
    total.backward()
    for network, optimizer, label in components:
        diag[label + "_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0))
        optimizer.step()
    diag.update(total_loss=float(total.detach()), critic_loss=float(critic_loss.detach()),
                actor_loss=float(actor_loss.detach()))
    return diag


def run_trial(config: dict, agent=None) -> dict:
    """Run one bounded trial, returning JSON-compatible observations only.

    Required config: problem, instance_path, method, seed, budget_mode,
    budget_value. Optional: checkpoint_path, rollout_size=8, log_stride=10,
    save_checkpoint_path (training only), device='cpu'. Time budgets count
    initialization. A candidate finished after the deadline never enters best.
    Each noninterruptible primitive may cause overshoot, explicitly reported.
    Unfinished rollout at a time cutoff is discarded; no post-budget training.
    """
    cfg = dict(config)
    problem, method = cfg["problem"], cfg["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    mode, budget = cfg["budget_mode"], float(cfg["budget_value"])
    if mode not in {"time", "iterations"} or budget <= 0:
        raise ValueError("Positive time or iteration budget required")
    if mode == "iterations" and int(budget) != budget:
        raise ValueError("Iteration budget must be an integer")
    if cfg.get("device", "cpu") != "cpu":
        raise ValueError("Controlled trials use CPU workers with one torch thread")
    if method in {"Frozen", "Adapt"} and not cfg.get("checkpoint_path") and agent is None:
        raise ValueError("Frozen and Adapt require a common checkpoint")
    if agent is not None and cfg.get("checkpoint_path"):
        raise ValueError("Use either persistent training agent or checkpoint_path")
    module = _module(problem)
    policy_module = _module(problem, True) if method == "AC" else module
    # The clock intentionally starts after process-global imports.
    start = time.perf_counter()
    elapsed = lambda: time.perf_counter() - start
    deadline = start + budget if mode == "time" else float("inf")
    times = {k: 0.0 for k in ("initialization", "graph", "inference", "operators", "training", "evaluation")}
    _seed(int(cfg["seed"]))
    model = module.Model()
    module.readCsvFile(str(cfg["instance_path"]), model)
    module.calDistanceMatrix(model)
    nodes = [model.depot] + [model.demand_dict[nid] for nid in model.node_id_list]
    sol = module.Sol()
    sol.node_no_seq = module.genInitialSol(model)
    if problem == "CVRP":
        sol.obj, sol.route_list, sol.route_distance = module.calObj(sol.node_no_seq, model)
    else:
        sol.obj, sol.route_list, sol.route_distance, sol.timetable_list = module.calObj(sol.node_no_seq, model)
    if not math.isfinite(sol.obj):
        raise ValueError("No feasible initial solution")
    # Split DP sums float32; use explicit route distance consistently from here.
    initial_dp_objective = float(sol.obj)
    sol.obj = float(sum(sol.route_distance))
    model.best_sol = copy.deepcopy(sol)
    initial_obj = float(sol.obj)
    initial_ready = elapsed()
    best_events = [{"time": initial_ready, "iteration": 0, "objective": initial_obj,
                    "route_count": len(sol.route_list)}] if time.perf_counter() <= deadline else []
    is_neural = method != "ALNS"
    training = is_neural and method != "Frozen"
    persistent = agent is not None
    if is_neural:
        base_edges, base_weights = module.precompute_base_graph(nodes, model, k_nn=5,
                                                               use_full_graph_knn=False, weight_mode="inv")
        if agent is None:
            # Seed has already been set and environment initialization is deterministic.
            cls = policy_module.SingleACAgent if method == "AC" else policy_module.DACAgent
            agent = cls(enc_dim=64, hid=64, gamma_repair=0.99, critic_coef=1.0).cpu()
            agent._controlled_problem = problem
            if cfg.get("checkpoint_path"):
                load_checkpoint(agent, cfg["checkpoint_path"])
        if getattr(agent, "_controlled_problem", problem) != problem:
            raise ValueError("Persistent agent problem mismatch")
        agent.train(training)
    times["initialization"] = elapsed()
    rollout, updates, iteration_log = [], [], []
    pair_counts = np.zeros((5, 5), dtype=np.int64)
    weights = np.ones(25, dtype=np.float64)
    entropy_sums = np.zeros(2)
    iterations, attempts, accepted_count, new_best_count = 0, 0, 0, 0
    stagnation_steps, last_improvement = 0, initial_ready
    stride = int(cfg.get("log_stride", 10))
    rollout_size = int(cfg.get("rollout_size", 8))
    if rollout_size <= 0 or stride <= 0:
        raise ValueError("rollout_size and log_stride must be positive")
    initial_weights_hash = None
    if is_neural and cfg.get("record_parameter_hash", False):
        initial_weights_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in agent.state_dict().values())).hexdigest()

    def progress(index):
        return min(1.0, elapsed() / budget) if mode == "time" else min(1.0, index / max(1.0, budget - 1.0))

    def state(graph, stage, ratio, did, p, gap, stagnation, temperature_ratio):
        return module.make_state_record(graph, stage, ratio, did, p, temperature_ratio, gap, stagnation)

    def graph(solution, removed=None):
        return module.build_graph(solution, nodes, model, base_edges, base_weights,
                                  removed_set=removed, include_path_edges=True)

    while time.perf_counter() < deadline and (mode == "time" or iterations < int(budget)):
        p = progress(iterations)
        temperature_progress = p if mode == "time" else iterations / budget
        temperature_ratio = 0.001 ** temperature_progress
        model.T0 = 50.0 * temperature_ratio
        gap = (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6)
        stagnation = min(1.0, (elapsed() - last_improvement) / budget) if mode == "time" else stagnation_steps / budget
        attempts += 1
        if is_neural:
            mark = time.perf_counter()
            full_graph = graph(sol)
            full_record = state(full_graph, 0, 0.0, -1, p, gap, stagnation, temperature_ratio)
            times["graph"] += time.perf_counter() - mark
            mark = time.perf_counter()
            with torch.no_grad():
                encoded = policy_module.encode_single_state(agent, full_record)
                if method == "AC":
                    joint_action, probabilities = agent.act_joint(encoded, deterministic=False)
                    ad, ar = divmod(joint_action, 5)
                    probs_matrix = probabilities.reshape(5, 5)
                    dp, rp = probs_matrix.sum(axis=1), probs_matrix.sum(axis=0)
                else:
                    ad, dp = agent.act_destroy(encoded, deterministic=False)
            times["inference"] += time.perf_counter() - mark
        else:
            mark = time.perf_counter()
            cumulative = (weights / weights.sum()).cumsum() - np.random.rand()
            selected = np.flatnonzero(cumulative > 0)
            joint_action = int(selected[0]) if len(selected) else 0
            ad, ar = divmod(joint_action, 5)
            times["inference"] += time.perf_counter() - mark
        if time.perf_counter() >= deadline:
            break
        mark = time.perf_counter()
        # Fractional iteration implements the same original removal schedule.
        destroy_progress = p if mode == "time" else iterations / budget
        destroyed, removed = module.doDestroy_by_id(ad, model, sol, destroy_progress, 1.0)
        times["operators"] += time.perf_counter() - mark
        if is_neural and method != "AC":
            mark = time.perf_counter()
            partial_sol = module.apply_destroy_to_partial_sol_explicit(destroyed, model)
            partial_graph = graph(partial_sol, set(removed))
            partial_record = state(partial_graph, 1, len(removed) / max(1, model.number_of_demands), ad, p, gap, stagnation, temperature_ratio)
            if method == "DAC-Pre":
                # This graph was constructed before destruction; it has no removal
                # mask, route-change feature, partial cost or removed-count signal.
                repair_record = state(full_graph, 1, 0.0, ad, p, gap, stagnation, temperature_ratio)
            else:
                repair_record = partial_record
            times["graph"] += time.perf_counter() - mark
            mark = time.perf_counter()
            with torch.no_grad():
                partial_encoded = module.encode_single_state(agent, repair_record)
                ar, rp = agent.act_repair(partial_encoded, deterministic=False)
            times["inference"] += time.perf_counter() - mark
        if time.perf_counter() >= deadline:
            break
        pair_counts[ad, ar] += 1
        mark = time.perf_counter()
        candidate = module.doRepair_by_id(ar, destroyed, removed, model)
        candidate_time = elapsed()
        times["operators"] += time.perf_counter() - mark
        if time.perf_counter() >= deadline:
            # Candidate objective is known only now; it is outside the budget.
            break
        mark = time.perf_counter()
        before, best_before = float(sol.obj), float(model.best_sol.obj)
        candidate_obj = float(candidate.obj)
        accepted = math.isfinite(candidate_obj) and (candidate_obj < before or
                   math.exp(min(0.0, (before - candidate_obj) / max(model.T0, 1e-6))) >= random.random())
        if accepted:
            sol = copy.deepcopy(candidate)
            accepted_count += 1
        found_best = candidate_obj < model.best_sol.obj
        if found_best:
            model.best_sol = copy.deepcopy(candidate)
            stagnation_steps = 0
            last_improvement = candidate_time
            new_best_count += 1
            best_events.append({"time": candidate_time, "iteration": iterations + 1,
                                "objective": candidate_obj, "route_count": len(candidate.route_list)})
        else:
            stagnation_steps += 1
        reward = module.calculate_transition_reward(before, candidate_obj, best_before,
                                                     float(model.best_sol.obj), initial_obj, accepted)
        if not math.isfinite(reward):
            raise FloatingPointError("Nonfinite transition reward")
        iterations += 1
        if not is_neural:
            index = ad * 5 + ar
            weights[index] = 0.9 * weights[index] + 0.1 * max(1e-6, reward + 1.0)
        else:
            entropy_sums += [-float(np.sum(dp * np.log(np.maximum(dp, 1e-30)))),
                            -float(np.sum(rp * np.log(np.maximum(rp, 1e-30))))]
        times["evaluation"] += time.perf_counter() - mark
        if training and time.perf_counter() < deadline:
            mark = time.perf_counter()
            next_graph = graph(sol) if accepted else full_graph
            npg = progress(iterations)
            next_stagnation = min(1.0, (elapsed() - last_improvement) / budget) if mode == "time" else stagnation_steps / budget
            next_temperature = 0.001 ** (npg if mode == "time" else iterations / budget)
            next_record = state(next_graph, 0, 0.0, -1, npg,
                                (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6), next_stagnation, next_temperature)
            done = float(mode == "iterations" and iterations >= int(budget))
            record = {"full": full_record, "ad": ad, "ar": ar, "reward": reward,
                      "next": next_record, "done": done}
            if method != "AC":
                record.update(partial=partial_record, repair_actor=repair_record)
            rollout.append(record)
            times["graph"] += time.perf_counter() - mark
            if (len(rollout) >= rollout_size or done) and time.perf_counter() < deadline:
                mark = time.perf_counter()
                es, ee = (0.01, 0.001) if problem == "VRPTW" else (0.05, 0.02)
                entropy = module.get_entropy_coef(p, entropy_start=es, entropy_end=ee)
                diagnostic = _update(agent, rollout, method, entropy)
                diagnostic.update(iteration=iterations, time=elapsed(), tail_batch=len(rollout) < rollout_size)
                updates.append(diagnostic)
                rollout.clear()
                times["training"] += time.perf_counter() - mark
        if iterations % stride == 0 or found_best or (mode == "iterations" and iterations == int(budget)):
            iteration_log.append({"iteration": iterations, "time": candidate_time,
                                  "best_objective": float(model.best_sol.obj), "current_objective": float(sol.obj),
                                  "candidate_objective": candidate_obj if math.isfinite(candidate_obj) else None,
                                  "destroy": int(ad), "repair": int(ar), "removed_count": len(removed),
                                  "reward": reward, "accepted": bool(accepted), "new_best": bool(found_best),
                                  "progress": p, "temperature": float(model.T0)})
    total = elapsed()
    times["other"] = max(0.0, total - sum(times.values()))
    anytime = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        cutoff = budget * fraction
        key = "time" if mode == "time" else "iteration"
        eligible = [e for e in best_events if e[key] <= cutoff]
        e = eligible[-1] if eligible else None
        anytime.append({"fraction": fraction, "budget": cutoff, "objective": e["objective"] if e else None,
                        "route_count": e["route_count"] if e else None})
    have_best = bool(best_events)
    final_hash = None
    if initial_weights_hash is not None:
        final_hash = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in agent.state_dict().values())).hexdigest()
    result = {"engine_version": ENGINE_VERSION, "config": cfg, "status": "ok" if have_best else "initialization_exceeded_budget",
              "problem": problem, "method": method, "seed": int(cfg["seed"]),
              "instance": Path(cfg["instance_path"]).stem, "customer_count": model.number_of_demands,
              "budget_mode": mode, "budget_value": budget, "elapsed_seconds": total,
              "overshoot_seconds": max(0.0, total - budget) if mode == "time" else 0.0,
              "initial_objective": initial_obj, "initial_split_objective": initial_dp_objective,
              "initial_solution_ready_seconds": initial_ready,
              "best_objective": float(model.best_sol.obj) if have_best else None,
              "route_count": len(model.best_sol.route_list) if have_best else None,
              "routes": [[int(n) for n in route] for route in model.best_sol.route_list] if have_best else [],
              "route_distances": [float(v) for v in model.best_sol.route_distance] if have_best else [],
              "iterations": iterations, "iteration_attempts": attempts,
              "accepted_count": accepted_count, "improvement_count": new_best_count,
              "updates": len(updates), "untrained_rollout_records": len(rollout),
              "destroy_counts": pair_counts.sum(axis=1).tolist(), "repair_counts": pair_counts.sum(axis=0).tolist(),
              "pair_counts": pair_counts.tolist(), "mean_action_entropy": (entropy_sums / max(1, iterations)).tolist(),
              "time_breakdown_seconds": times, "anytime": anytime, "best_events": best_events,
              "iteration_log": iteration_log, "update_diagnostics": updates,
              "persistent_training_agent": persistent,
              "initial_parameter_sha256": initial_weights_hash, "final_parameter_sha256": final_hash,
              "protocol_notes": ["Archived DAC environment for all methods; explicit route distance objective",
                                 "CPU one Torch thread; imports/JIT warmup excluded, trial initialization included",
                                 "Wall-clock progress drives temperature/removal/entropy schedules in time mode",
                                 "Candidate completed after deadline discarded; primitive overshoot reported",
                                 "No final gradient update after time limit; pending rollout count reported"]}
    if cfg.get("save_checkpoint_path"):
        if not is_neural:
            raise ValueError("ALNS has no checkpoint")
        result["saved_checkpoint_sha256"] = save_checkpoint(agent, cfg["save_checkpoint_path"],
                                                            {"training_config": cfg, "iterations": iterations})
    return result
