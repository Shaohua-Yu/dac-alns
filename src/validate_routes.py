"""Independent, float64 route checks; no imports from the search implementation.

Default IDs match the original solvers: depot 0, customers 1..n.
Both [customer,...] and [0,customer,...,0] are accepted. A dict of
{'routes': ..., 'id_space': 'source'} explicitly requests original CSV IDs.
Vehicle count is reported, not constrained, in the controlled distance protocol.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


def _value(row, names, default=0.0):
    for name in names:
        if str(row.get(name, '')).strip():
            return float(row[name])
    return float(default)


def read_instance(instance_path, problem):
    problem = problem.upper()
    if problem not in {'CVRP', 'VRPTW'}:
        raise ValueError(f'Unknown problem: {problem}')
    with Path(instance_path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty instance')
    candidates = [r for r in rows if int(float(r['id'])) == 0 or
                  str(r.get('is_depot', '')).lower() in {'1', 'true', 'yes', 'y'}]
    if not candidates:
        candidates = [r for r in rows if _value(r, ('demand',)) == 0]
    if len(candidates) != 1:
        raise ValueError('Exactly one depot is required')
    depot = candidates[0]
    ordered = [depot] + [r for r in rows if r is not depot]
    source_ids = [int(float(r['id'])) for r in ordered]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('Duplicate CSV source IDs')
    nodes = {}
    for index, row in enumerate(ordered):
        internal = 0 if index == 0 else (int(float(row['id'])) - 1 if problem == 'CVRP' else index)
        if index and internal == 0:
            raise ValueError('CVRP source IDs incompatible with original id-1 convention')
        nodes[internal] = {
            'source_id': int(float(row['id'])),
            'x': _value(row, ('x_coord',)), 'y': _value(row, ('y_coord',)),
            'demand': 0.0 if index == 0 else _value(row, ('demand',)),
            'ready': _value(row, ('depot_start_time', 'ready_time', 'start_time') if index == 0 else ('ready_time', 'start_time')),
            'due': _value(row, ('depot_end_time', 'due_time', 'due_date', 'end_time') if index == 0 else ('due_time', 'due_date', 'end_time'), math.inf),
            'service': 0.0 if index == 0 else _value(row, ('service_time', 'service')),
        }
    if len(nodes) != len(rows):
        raise ValueError('Duplicate internal IDs')
    return {
        'nodes': nodes, 'n': len(nodes)-1,
        'capacity': _value(depot, ('v_cap', 'capacity'), 100.0),
        'speed': _value(depot, ('v_speed', 'speed'), 1.0),
        'max_vehicles': _value(depot, ('v_num', 'max_vehicles'), math.inf),
        'problem': problem,
    }


def validate_routes(instance_path, problem, routes, *, tolerance=1e-6):
    """Return valid, raw Euclidean distance, n_routes, errors and route evidence.

    Start time is the depot ready time; travel time is raw Euclidean distance /
    speed; waiting is allowed; service must start within the customer's window;
    return to depot must precede its closing time. Empty routes are ignored.
    """
    result = {'valid': False, 'distance': 0.0, 'n_routes': 0, 'errors': [],
              'id_space': 'internal', 'depot_internal_id': 0,
              'distance_rule': 'raw_float64_euclidean',
              'fleet_constraint_enforced': False, 'tolerance': tolerance,
              'normalized_routes': [], 'source_id_routes': [], 'route_details': []}
    errors = result['errors']
    try:
        data = read_instance(instance_path, problem)
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f'instance_parse: {exc}')
        return result
    nodes = data['nodes']
    if data['speed'] <= 0:
        errors.append('speed must be positive')
        return result
    source_to_internal = {node['source_id']: key for key, node in nodes.items()}
    result['internal_to_source'] = {str(key): node['source_id'] for key, node in nodes.items()}
    id_space = 'internal'
    if isinstance(routes, dict):
        id_space = routes.get('id_space', 'internal')
        routes = routes.get('routes', [])
    if id_space not in {'internal', 'source'}:
        errors.append(f'unknown id_space: {id_space}')
        return result
    if routes is None:
        routes = []
    visited = []
    def distance(a, b):
        return math.hypot(nodes[a]['x']-nodes[b]['x'], nodes[a]['y']-nodes[b]['y'])
    for route_index, raw_route in enumerate(routes):
        try:
            route = []
            for value in raw_route:
                integer = int(value)
                if float(value) != integer:
                    raise ValueError(f'noninteger node {value}')
                route.append(source_to_internal[integer] if id_space == 'source' else integer)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f'route {route_index}: malformed/unknown ID: {exc}')
            continue
        if route and route[0] == 0:
            route = route[1:]
        if route and route[-1] == 0:
            route = route[:-1]
        if not route:
            continue
        if 0 in route:
            errors.append(f'route {route_index}: depot appears within route')
        customers = [node for node in route if node != 0]
        visited.extend(customers)
        invalid = [node for node in route if node not in nodes]
        if invalid:
            errors.append(f'route {route_index}: unknown internal IDs {invalid}')
            continue
        full_route = [0] + route + [0]
        result['normalized_routes'].append(full_route)
        result['source_id_routes'].append([nodes[node]['source_id'] for node in full_route])
        result['n_routes'] += 1
        length = sum(distance(a, b) for a, b in zip(full_route, full_route[1:]))
        load = sum(nodes[node]['demand'] for node in customers)
        result['distance'] += length
        if load > data['capacity'] + tolerance:
            errors.append(f'route {route_index}: capacity {load:.12g}>{data["capacity"]:.12g}')
        detail = {'route_index': route_index, 'distance': length, 'load': load}
        if data['problem'] == 'VRPTW':
            current_time = nodes[0]['ready']
            previous = 0
            timings = []
            for customer in customers:
                node = nodes[customer]
                arrival = current_time + distance(previous, customer) / data['speed']
                start = max(arrival, node['ready'])
                if start > node['due'] + tolerance:
                    errors.append(f'route {route_index}, node {customer}: late service {start:.12g}>{node["due"]:.12g}')
                current_time = start + node['service']
                timings.append({'node': customer, 'arrival': arrival, 'service_start': start,
                                'departure': current_time})
                previous = customer
            returned = current_time + distance(previous, 0) / data['speed']
            if returned > nodes[0]['due'] + tolerance:
                errors.append(f'route {route_index}: late depot return {returned:.12g}>{nodes[0]["due"]:.12g}')
            detail.update({'return_time': returned, 'timings': timings})
        result['route_details'].append(detail)
    counts = Counter(visited)
    expected = set(nodes) - {0}
    missing = sorted(expected - counts.keys())
    duplicated = {str(node): count for node, count in counts.items() if count != 1}
    if missing:
        errors.append(f'missing customers: {missing}')
    if duplicated:
        errors.append(f'customers not visited exactly once: {duplicated}')
    if not math.isfinite(result['distance']):
        errors.append('nonfinite objective')
    result['max_vehicles_in_data'] = None if math.isinf(data['max_vehicles']) else data['max_vehicles']
    result['fleet_limit_exceeded'] = result['n_routes'] > data['max_vehicles']
    result['valid'] = not errors
    return result
