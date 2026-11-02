"""Time-only DR-ALNS adaptation: no step limit, no restarts.
Native progress k/1000 is capped at1 after the training horizon.
Operators, checkpoint, acceptance, RNG and initialization match archived E4.
"""
import copy
import importlib
import pickle
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from external_common import BASE, read_instance, validate, original_routes, finish, sha256, normalize_config

COMMIT = '144e48f6912ad9967968561b6b0ce90cedcc14aa'
UPSTREAM = BASE / 'vendor' / ('DR-ALNS-' + COMMIT)


def load_modules():
    # Wang vendors a different package named alns. Run these methods in separate
    # subprocesses to preserve each official package unchanged.
    sys.path.insert(0, str(UPSTREAM / 'code/src'))
    from rl.environments.cvrp_AlnsEnv_LSA1 import cvrpAlnsEnv_LSA1
    from stable_baselines3 import PPO
    import gymnasium as gym
    return cvrpAlnsEnv_LSA1, PPO, gym


def run_trial(config):
    config = normalize_config(config)
    if 'budget_s' not in config and 'budget_value' in config:
        config['budget_s'] = config['budget_value']
    Env, PPO, gym = load_modules()
    torch.set_num_threads(1)
    seed = int(config.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    instance = read_instance(config['instance_path'])
    n = len(instance['ids']) - 1
    # Pre-declared nearest training size, never selected on test performance.
    model_n = min([20, 50, 100], key=lambda v: (abs(v - n), v))
    checkpoint = Path(config.get('checkpoint', UPSTREAM / 'code/src/rl/trained_models/rl/cvrpAlnsEnv_LSA1/models' / f'DR-ALNS_CVRP_{model_n}/model.zip'))
    checkpoint_seconds_start = time.perf_counter()
    # Published archive uses old gym space pickles; dimensions remain unchanged.
    model = PPO.load(str(checkpoint), device='cpu', custom_objects={
        'observation_space': gym.spaces.Box(0, 100, shape=(8,), dtype=np.float64),
        'action_space': gym.spaces.MultiDiscrete([3, 1, 10, 100])})
    checkpoint_seconds = time.perf_counter() - checkpoint_seconds_start
    # CSV conversion creates a one-instance native file; upstream reader unchanged.
    folder = BASE / 'data/external_dr_inputs'
    folder.mkdir(parents=True, exist_ok=True)
    instance_file = folder / (sha256(config['instance_path']) + '.pkl')
    if not instance_file.exists():
        # Unique task filenames avoid concurrent readers of a partial file.
        import os
        temp = folder / (instance_file.name + f'.{os.getpid()}.tmp')
        with temp.open('wb') as f:
            pickle.dump([(instance['xy'][0], instance['xy'][1:], instance['demand'][1:], instance['capacity'])], f)
        temp.replace(instance_file)
    horizon = int(config.get('native_iterations', 1000))
    env = Env({'environment': {'instance_file': str(instance_file.resolve()), 'instance_nr': [0], 'iterations': horizon}})
    obs, _ = env.reset(seed=seed)
    env.rnd_state = np.random.RandomState(seed)
    best_routes = copy.deepcopy(env.best_solution.routes)
    best_obj = float(env.best_solution.objective())
    initial_obj = best_obj
    budget = config.get('budget_s')
    deadline = float('inf') if budget is None else t0 + float(budget)
    if budget is None or float(budget) <= 0: raise ValueError('Positive time budget required')
    trace = [{'iteration': 0, 'elapsed_s': time.perf_counter() - t0, 'best_obj': best_obj, 'route_count':len(best_routes)}]
    iteration = 0
    overshot_candidate = False
    inference_seconds = operator_seconds = 0.
    while time.perf_counter() < deadline:
        ts = time.perf_counter()
        action, _ = model.predict(obs, deterministic=bool(config.get('deterministic', False)))
        inference_seconds += time.perf_counter() - ts
        # Upstream step() writes an absolute HPC path on horizon. Keep its
        # search-progress horizon for observation while disabling that I/O-only
        # endpoint by extending max_iterations during step, then rebuild obs.
        env.max_iterations = float('inf')
        ts = time.perf_counter()
        obs, reward, done, truncated, info = env.step(action)
        operator_seconds += time.perf_counter() - ts
        env.max_iterations = horizon
        obs = env.make_observation()
        obs[5] = min(1., iteration / horizon + 1. / horizon)
        candidate_completed = time.perf_counter()
        if candidate_completed > deadline:
            overshot_candidate = True
            break
        iteration += 1
        obj = float(env.best_solution.objective())
        if obj < best_obj:
            best_obj = obj
            best_routes = copy.deepcopy(env.best_solution.routes)
        trace.append({'iteration': iteration, 'elapsed_s': candidate_completed - t0,
                      'best_obj': best_obj, 'route_count':len(best_routes), 'current_obj': float(env.current_solution.objective()),
                      'actions': [int(x) for x in action], 'reward': float(reward)})
    elapsed = time.perf_counter() - t0
    best_routes = [[int(v) for v in r] for r in best_routes]
    result = {'algorithm': 'DR-ALNS-time-only', 'upstream_commit': COMMIT,
              'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
              'checkpoint_training_customers': model_n, 'objective': best_obj,
              'initial_objective': initial_obj, 'elapsed_s': elapsed, 'iterations': iteration,
              'routes': [[0] + r + [0] for r in best_routes], 'raw_id_routes': original_routes(best_routes, instance),
              'route_indexing': 'CSV row position, depot zero, endpoints included',
              'deadline_overrun_s': max(0., elapsed - float(budget)) if budget is not None else 0.,
              'overshot_candidate_excluded': overshot_candidate,
              'native_iteration_limit': None, 'original_training_horizon': horizon,
              'stopping_reason': 'time_limit', 'restart_policy': 'continuous search without restart',
              'progress_policy': 'min(iterations/1000,1); retained original progress up to1000',
              'stopping_adaptation': 'removed1000-step cap; only wall-clock deadline',
              'timing': {'checkpoint_s': checkpoint_seconds, 'inference_s': inference_seconds, 'operators_s': operator_seconds},
              'compatibility_changes': ['CSV to native pickle', 'seed native RNG', 'gym to gymnasium checkpoint spaces',
                                        'external deadline', 'redirect trajectory via wrapper; skip hardcoded HPC endpoint'],
              'distance_metric': 'unrounded Euclidean', 'operator_library': 'official 3 destroy / 1 regret repair'}
    result.update(validate(best_routes, instance))
    return finish(result, trace, config)
