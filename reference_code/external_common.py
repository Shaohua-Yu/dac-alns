"""Independent input/route audit for external baselines (unrounded Euclidean)."""
from pathlib import Path
import csv
import json
import math
import hashlib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent


def normalize_config(config):
    config = dict(config)
    if config.get('budget_mode') == 'iterations':
        config['budget_s'] = None
        config['max_iterations'] = int(config['budget_value'])
    elif 'budget_s' not in config and 'budget_value' in config:
        config['budget_s'] = float(config['budget_value'])
    if config.get('checkpoint_path') and not config.get('checkpoint'):
        config['checkpoint'] = config['checkpoint_path']
    return config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_instance(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))
    ids = [int(r['id']) for r in rows]
    xy = np.asarray([[float(r['x_coord']), float(r['y_coord'])] for r in rows])
    demand = np.asarray([float(r['demand']) for r in rows])
    depot = int(np.flatnonzero(demand == 0)[0])
    order = [depot] + [i for i in range(len(rows)) if i != depot]
    xy, demand = xy[order], demand[order]
    ids = [ids[i] for i in order]
    distance = np.sqrt(np.sum((xy[:, None] - xy[None, :]) ** 2, axis=-1))
    return {'ids': ids, 'xy': xy, 'demand': demand, 'distance': distance,
            'capacity': float(rows[depot].get('v_cap', rows[depot].get('vehicle_capacity', 0))),
            'rows': [rows[i] for i in order]}


def validate(routes, instance, tw=False):
    """Routes use matrix indices with depot omitted."""
    n = len(instance['ids']) - 1
    flat = [int(v) for r in routes for v in r]
    errors = []
    if sorted(flat) != list(range(1, n + 1)):
        errors.append('customer coverage/duplicates')
    total = 0.
    for r in routes:
        if not r:
            errors.append('empty route')
            continue
        if sum(instance['demand'][i] for i in r) > instance['capacity'] + 1e-6:
            errors.append('capacity')
        nodes = [0] + list(r) + [0]
        total += sum(instance['distance'][a, b] for a, b in zip(nodes, nodes[1:]))
        if tw:
            t = 0.
            for a, b in zip(nodes, nodes[1:]):
                t += instance['distance'][a, b]
                t = max(t, instance['time_windows'][b, 0])
                if t > instance['time_windows'][b, 1] + 1e-6:
                    errors.append('time window')
                t += instance['service_times'][b]
    return {'feasible': not errors, 'validation_errors': sorted(set(errors)),
            'verified_objective': float(total), 'route_count': len(routes)}


def original_routes(routes, instance):
    ids = instance['ids']
    return [[ids[0]] + [ids[i] for i in r] + [ids[0]] for r in routes]


def finish(result, trace, config):
    result['trace'] = trace
    result['instance_path'] = str(Path(config['instance_path']).resolve())
    result['instance_sha256'] = sha256(config['instance_path'])
    result['seed'] = int(config.get('seed', 0))
    result['budget_s'] = config.get('budget_s')
    result['elapsed_seconds'] = result['elapsed_s']
    result['best_objective'] = result['objective']
    result['overshoot_seconds'] = result.get('deadline_overrun_s', 0.)
    mode = config.get('budget_mode', 'time' if config.get('budget_s') is not None else 'iterations')
    budget = float(config.get('budget_value') or config.get('budget_s') or config.get('max_iterations') or result['iterations'])
    events = []
    for row in trace:
        within = row['elapsed_s'] <= budget if mode == 'time' else row['iteration'] <= budget
        if within and (not events or row['best_obj'] < events[-1]['objective']):
            events.append({'time':row['elapsed_s'], 'iteration':row['iteration'],
                           'objective':row['best_obj'], 'route_count':row.get('route_count')})
    result['best_events'] = events
    anytime = []
    for fraction in (.1, .25, .5, .75, 1.):
        cutoff = budget * fraction
        eligible = [e for e in events if e['time' if mode == 'time' else 'iteration'] <= cutoff]
        event = eligible[-1] if eligible else None
        anytime.append({'fraction':fraction, 'budget':cutoff,
                        'objective':event['objective'] if event else None,
                        'route_count':event['route_count'] if event else None})
    result['anytime'] = anytime
    result['status'] = 'ok' if events else 'initialization_exceeded_budget'
    if not events:
        result['initialization_diagnostic_objective'] = result['objective']
        result['initialization_diagnostic_routes'] = result['routes']
        result.update(objective=None, best_objective=None, routes=[], route_count=None)
    if config.get('trace_path'):
        p = Path(config['trace_path'])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in trace), encoding='utf8')
        result['trace_path'] = str(p.resolve())
    return result
