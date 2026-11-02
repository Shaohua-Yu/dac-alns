"""Wang official 5x3 environment; equivalent normalized CSV adapter.

Must execute in a dedicated process: upstream vendors its own `alns` package.
Time-only adaptation: native done signals are recorded but do not end search.
Native actions/operators/acceptance unchanged; progress is capped at1 after100steps.
"""
import copy
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from external_common import BASE, read_instance, validate, original_routes, finish, sha256, normalize_config

COMMIT = '64c0ba11902c7db805f64175b2db2dcfb58df7ad'
UPSTREAM = BASE / 'vendor' / ('ppo-alns-' + COMMIT)
sys.path.insert(0, str(UPSTREAM))
from ppo import vrptwenv as native
from core import VRPTWGeneratorR
from stable_baselines3 import PPO


class FixedPool:
    def __init__(self, data):
        self.data = data

    def sample(self):
        return self.data


class SeededEnvironment(native.VRPTWEnvironment):
    """Fix upstream's entropy-seeded default_rng, without changing the algorithm."""
    def __init__(self, pool, seed):
        self.reset_seed = int(seed)
        self.reset_count = 0
        super().__init__(pool)

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.alns._rng = np.random.default_rng(self.reset_seed + self.reset_count)
        self.reset_count += 1
        return result


def csv_data(path):
    instance = read_instance(path)
    rows = instance['rows']
    tw = np.asarray([[float(r['ready_time']), float(r['due_date'])] for r in rows])
    service = np.asarray([float(r['service_time']) for r in rows])
    instance['time_windows'] = tw
    instance['service_times'] = service
    horizon = float(tw[0, 1])
    # Native generator uses travel_times=edge_weight/50 and capacity=1.
    # This is a uniform unit conversion of the test problem, not a new solver.
    scale = horizon / 50.
    data = {'dimension': len(rows), 'capacity': 1., 'num_vehicles': len(rows) - 1,
            'max_travel_time': 1., 'depot': 0, 'demand': instance['demand'] / instance['capacity'],
            'time_windows': tw / horizon, 'service_times': service / horizon,
            'travel_times': instance['distance'] / horizon,
            'node_coord': instance['xy'] / scale, 'edge_weight': instance['distance'] / scale}
    # Native CvrptwState is initialized with travel_times (not edge_weight).
    return instance, data, horizon


def run_trial(config):
    config = normalize_config(config)
    if 'budget_s' not in config and 'budget_value' in config:
        config['budget_s'] = config['budget_value']
    checkpoint = Path(config['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError('Official Wang repository ships no checkpoint; complete independent training first.')
    torch.set_num_threads(1)
    seed = int(config.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    instance, data, scale = csv_data(config['instance_path'])
    native.ppo_config.dimension = len(instance['ids'])
    env = SeededEnvironment(FixedPool(data), seed)
    state = env.state
    best_routes = copy.deepcopy(env.best_solution.routes)
    best_obj = float(env.best_solution.objective()) * scale
    initial_obj = best_obj
    trace = [{'iteration': 0, 'elapsed_s': time.perf_counter() - t0, 'best_obj': best_obj, 'route_count':len(best_routes)}]
    model = PPO.load(str(checkpoint), device='cpu')
    # Predict checks dimensionality and prevents applying a wrong-size model.
    budget = config.get('budget_s')
    deadline = float('inf') if budget is None else t0 + float(budget)
    if budget is None or float(budget) <= 0: raise ValueError('Positive time budget required')
    native_done_signals = stop_action_count = accept_zero_count = accept_one_count = 0
    first_native_done_iteration = None
    iteration = 0
    done = False
    overshot = False
    while time.perf_counter() < deadline:
        action, _ = model.predict(state, deterministic=bool(config.get('deterministic', True)))
        state, reward, done, truncated, info = env.step(action)
        native_done_signals += int(done)
        stop_action_count += int(action[3])
        accept_zero_count += int(action[2] == 0)
        accept_one_count += int(action[2] == 1)
        if done and first_native_done_iteration is None: first_native_done_iteration = iteration + 1
        state['search_progress'][0] = min(1., env.iters / native.MAX_ITERATIONS)
        candidate_completed = time.perf_counter()
        if candidate_completed > deadline:
            overshot = True
            break
        iteration += 1
        obj = float(env.best_solution.objective()) * scale
        if obj < best_obj:
            best_obj = obj
            best_routes = copy.deepcopy(env.best_solution.routes)
        trace.append({'iteration': iteration, 'elapsed_s': candidate_completed - t0,
                      'best_obj': best_obj, 'route_count':len(best_routes), 'actions': [int(x) for x in action], 'reward': float(reward)})
    elapsed = time.perf_counter() - t0
    best_routes = [[int(v) for v in r] for r in best_routes]
    result = {'algorithm': 'Wang-PPO-ALNS-time-only', 'upstream_commit': COMMIT,
              'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
              'objective': best_obj, 'initial_objective': initial_obj, 'elapsed_s': elapsed,
              'iterations': iteration, 'routes': [[0] + r + [0] for r in best_routes],
              'raw_id_routes': original_routes(best_routes, instance),
              'route_indexing': 'CSV row position, depot zero, endpoints included',
              'deadline_overrun_s': max(0., elapsed - float(budget)) if budget is not None else 0.,
              'overshot_candidate_excluded': overshot, 'native_early_stop': False,
              'stopping_reason': 'time_limit', 'native_done_signals_ignored': native_done_signals,
              'first_native_done_iteration': first_native_done_iteration, 'stop_actions_ignored': stop_action_count,
              'accept_zero_actions_attempted': accept_zero_count, 'accept_one_actions_attempted': accept_one_count,
              'native_iteration_limit': None, 'original_training_horizon': native.MAX_ITERATIONS,
              'restart_policy': 'continuous search without restart',
              'progress_policy': 'min(iterations/100,1); retained original progress up to100',
              'stopping_adaptation': 'ignore native done including learned stop and100-step cap; only wall-clock deadline',
              'operator_library': 'official 5 destroy / 3 repair',
              'distance_scale': scale, 'compatibility_changes': ['equivalent unit normalization', 'explicit seed for default_rng']}
    result.update(validate(best_routes, instance, tw=True))
    return finish(result, trace, config)
