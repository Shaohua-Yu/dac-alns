"""Deadline wrapper around recovered historical 7D, 5x5x10x50 online PPO.

The source algorithm/operators/update remain in the hashed vendor snapshot.
This wrapper adds the equal-time stopping rule and full feasible-route evidence.
"""
import copy
import importlib.util
import math
import sys
import time
import numpy as np
import torch
from external_common import BASE, read_instance, validate, original_routes, finish, sha256, normalize_config

SOURCE = BASE / 'vendor/historical_DRL_PPO_CVRP.py'
sys.path.insert(0, str(SOURCE.parent))
import historical_DRL_PPO_CVRP as h
h.DEVICE = torch.device('cpu')
torch.set_num_threads(1)
h._split_dp_core(np.asarray([1], dtype=np.int32), np.asarray([1.], dtype=np.float32),
                 2., np.asarray([[0., 1.], [1., 0.]], dtype=np.float32), depot_idx=0)


def run_trial(config):
    config = normalize_config(config)
    if 'budget_s' not in config and config.get('budget_type', 'time') == 'time' and 'budget_value' in config:
        config['budget_s'] = config['budget_value']
    h.set_seed(int(config.get('seed', 0)))
    t0 = time.perf_counter()
    budget = config.get('budget_s')
    deadline = float('inf') if budget is None else t0 + float(budget)
    instance = read_instance(config['instance_path'])
    model = h.Model()
    h.readCsvFile(str(config['instance_path']), model)
    h.calDistanceMatrix(model)
    current = h.initialize_solution(model)
    model.best_sol = copy.deepcopy(current)
    initial_ready = time.perf_counter() - t0
    original_iterations = int(config.get('original_iterations', max(50 * model.number_of_demands, 2000)))
    max_iterations = config.get('max_iterations')
    if budget is None and max_iterations is None:
        max_iterations = original_iterations
    agent = h.ReijnenStylePPO().to(h.DEVICE)
    agent.train()
    rollout = h.RolloutBuffer()
    state = np.asarray([0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)
    stagnation = updates = gradient_steps = iteration = 0
    state_seconds = inference_seconds = training_seconds = operator_seconds = 0.
    trace = [{'iteration': 0, 'elapsed_s': initial_ready, 'best_obj': float(current.obj), 'route_count':len(current.route_list)}]
    counts = [np.zeros(x, dtype=int) for x in h.ACTION_DIMS]
    initial_obj = float(current.obj)
    overshot_candidate = False
    while time.perf_counter() < deadline and (max_iterations is None or iteration < int(max_iterations)):
        ts = time.perf_counter()
        with torch.no_grad():
            actions, old_log_prob, probs = agent.act(state)
        inference_seconds += time.perf_counter() - ts
        d, r, severity, temperature_action = map(int, actions)
        ratio = h.destruction_ratio_from_action(severity)
        temperature = h.temperature_from_action(temperature_action)
        ts = time.perf_counter()
        destroyed, removed = h.destroy_with_ratio(d, ratio, model, current)
        candidate = h.doRepair(r, destroyed, removed, model, current)
        operator_seconds += time.perf_counter() - ts
        candidate_completed = time.perf_counter()
        if candidate_completed > deadline:
            overshot_candidate = True
            break
        iteration += 1
        for head, action in enumerate(actions):
            counts[head][int(action)] += 1
        improved = candidate.obj < current.obj
        new_best = candidate.obj < model.best_sol.obj
        accepted = improved or new_best or h.random.random() < math.exp((current.obj - candidate.obj) / max(temperature, 1e-9))
        if accepted:
            current = copy.deepcopy(candidate)
        if new_best:
            model.best_sol = copy.deepcopy(candidate)
            stagnation = 0
        else:
            stagnation += 1
        reward = h.NEW_BEST_REWARD if new_best else 0.
        ts = time.perf_counter()
        next_state = h.build_search_state(new_best, accepted, improved,
            math.isclose(current.obj, model.best_sol.obj, rel_tol=1e-9, abs_tol=1e-9),
            current.obj, model.best_sol.obj, stagnation, min(iteration, original_iterations), original_iterations)
        state_seconds += time.perf_counter() - ts
        done = float(max_iterations is not None and iteration == int(max_iterations))
        rollout.add(state, actions, old_log_prob, reward, next_state, done)
        state = next_state
        trace.append({'iteration': iteration, 'elapsed_s': candidate_completed - t0,
                      'best_obj': float(model.best_sol.obj), 'route_count':len(model.best_sol.route_list), 'candidate_obj': float(candidate.obj),
                      'current_obj': float(current.obj), 'accepted': bool(accepted), 'new_best': bool(new_best),
                      'actions': [int(a) for a in actions], 'removed_customers': len(removed)})
        if (len(rollout) >= 8 or done) and time.perf_counter() < deadline:
            ts = time.perf_counter()
            stats = agent.update(rollout)
            training_seconds += time.perf_counter() - ts
            updates += 1
            gradient_steps += int(stats['optimizer_steps'])
    elapsed = time.perf_counter() - t0
    routes = [[model.id2idx[int(v)] for v in r if int(v) != model.depot.id] for r in model.best_sol.route_list]
    result = {'algorithm': 'PPO-Adapted', 'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
              'objective': float(model.best_sol.obj), 'initial_objective': initial_obj,
              'routes': [[0] + r + [0] for r in routes], 'raw_id_routes': original_routes(routes, instance),
              'route_indexing': 'CSV row position, depot zero, endpoints included', 'elapsed_s': elapsed, 'iterations': iteration,
              'deadline_overrun_s': max(0., elapsed - float(budget)) if budget is not None else 0.,
              'overshot_candidate_excluded': overshot_candidate,
              'online_updates': updates, 'gradient_steps': gradient_steps,
              'timing': {'state_s': state_seconds, 'inference_s': inference_seconds,
                         'training_s': training_seconds, 'operators_s': operator_seconds},
              'action_counts': [x.tolist() for x in counts], 'state_progress_rule': 'min(iter/original_iterations,1)'}
    result.update(validate(routes, instance))
    return finish(result, trace, config)
