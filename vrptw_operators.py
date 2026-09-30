# -*- coding: utf-8 -*-
"""Shared, behaviorally distinct 5x5 operator library for hard VRPTW.

The module intentionally depends only on the common Model/Node field contract used
by the ALNS, Single-AC, and DAC programs.  It owns route feasibility and insertion
evaluation so every selector observes exactly the same operator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import random as _random
from typing import Any, Iterable, Sequence


EPS = 1e-9
FEASIBILITY_TOL = 1e-7


@dataclass(frozen=True)
class RouteEvaluation:
    feasible: bool
    distance: float
    load: float
    arrivals: tuple[float, ...]
    service_starts: tuple[float, ...]
    departures: tuple[float, ...]
    waiting_times: tuple[float, ...]
    total_waiting: float
    min_normalized_slack: float
    finish_time: float


@dataclass(frozen=True)
class InsertionOption:
    customer: int
    route_idx: int
    position: int
    is_new_route: bool
    distance_delta: float
    waiting_delta: float
    total_downstream_delay: float
    max_downstream_delay: float
    slack_loss: float
    min_slack_after: float


def _rng_or_default(rng: Any | None) -> Any:
    return rng if rng is not None else _random


def _depot_id(model: Any) -> int:
    return int(model.depot.id)


def _node(model: Any, customer: int) -> Any:
    return model.demand_dict[customer]


def _distance(model: Any, first: int, second: int) -> float:
    return float(
        model.distance_matrix[
            model.id2idx[first],
            model.id2idx[second],
        ]
    )


def _travel_time(model: Any, first: int, second: int) -> float:
    return float(
        model.time_matrix[
            model.id2idx[first],
            model.id2idx[second],
        ]
    )


def _infeasible_route() -> RouteEvaluation:
    return RouteEvaluation(
        feasible=False,
        distance=float("inf"),
        load=float("inf"),
        arrivals=(),
        service_starts=(),
        departures=(),
        waiting_times=(),
        total_waiting=float("inf"),
        min_normalized_slack=-float("inf"),
        finish_time=float("inf"),
    )


def evaluate_route(route: Sequence[int], model: Any) -> RouteEvaluation:
    """Evaluate capacity and hard time-window feasibility for one explicit route."""
    depot = _depot_id(model)
    if len(route) < 2 or route[0] != depot or route[-1] != depot:
        return _infeasible_route()

    clock = float(model.depot_start_time)
    load = 0.0
    distance = 0.0
    arrivals = [clock]
    service_starts = [clock]
    departures = [clock]
    waiting_times = [0.0]
    normalized_slacks: list[float] = []

    for position, (previous, current) in enumerate(zip(route, route[1:]), 1):
        distance += _distance(model, previous, current)
        arrival = clock + _travel_time(model, previous, current)

        if current == depot:
            if position != len(route) - 1:
                return _infeasible_route()
            if arrival > float(model.depot_end_time) + FEASIBILITY_TOL:
                return _infeasible_route()
            start = arrival
            departure = arrival
            waiting = 0.0
        else:
            node = _node(model, current)
            load += float(node.demand)
            start = max(arrival, float(node.start_time))
            if (
                load > float(model.vehicle_cap) + FEASIBILITY_TOL
                or start > float(node.end_time) + FEASIBILITY_TOL
            ):
                return _infeasible_route()
            departure = start + float(node.service_time)
            waiting = max(0.0, start - arrival)
            width = max(float(node.end_time - node.start_time), EPS)
            normalized_slacks.append(
                max(0.0, float(node.end_time) - start) / width
            )

        arrivals.append(arrival)
        service_starts.append(start)
        departures.append(departure)
        waiting_times.append(waiting)
        clock = departure

    return RouteEvaluation(
        feasible=True,
        distance=distance,
        load=load,
        arrivals=tuple(arrivals),
        service_starts=tuple(service_starts),
        departures=tuple(departures),
        waiting_times=tuple(waiting_times),
        total_waiting=sum(waiting_times),
        min_normalized_slack=(
            min(normalized_slacks) if normalized_slacks else 1.0
        ),
        finish_time=clock,
    )


def _active_route_copies(routes: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(route) for route in routes if len(route) > 2]


def _clean_routes(routes: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(route) for route in routes if len(route) > 2]


def _unique_customers(customers: Iterable[int], depot: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for customer in customers:
        customer = int(customer)
        if customer == depot or customer in seen:
            continue
        seen.add(customer)
        result.append(customer)
    return result


def _rank01(values: Sequence[float]) -> list[float]:
    """Return tie-preserving ordinal ranks, where the smallest value is zero."""
    if not values:
        return []
    unique = sorted(set(float(value) for value in values))
    if len(unique) == 1:
        return [0.0 for _ in values]
    denominator = float(len(unique) - 1)
    ranks = {value: index / denominator for index, value in enumerate(unique)}
    return [ranks[float(value)] for value in values]


def _biased_pick(sorted_items: Sequence[Any], rng: Any, exponent: float) -> Any:
    if not sorted_items:
        raise ValueError("Cannot select from an empty candidate list.")
    index = int(math.floor(len(sorted_items) * (rng.random() ** exponent)))
    return sorted_items[min(index, len(sorted_items) - 1)]


def enumerate_feasible_insertions(
    customer: int,
    routes: Sequence[Sequence[int]],
    model: Any,
) -> list[InsertionOption]:
    """Enumerate every hard-feasible existing-route position plus a new route."""
    options: list[InsertionOption] = []

    for route_idx, source_route in enumerate(routes):
        options.extend(_enumerate_route_insertions(
            customer,
            route_idx,
            source_route,
            model,
        ))

    singleton = _singleton_insertion_option(customer, model)
    if singleton is not None:
        options.append(singleton)

    return options


def _enumerate_route_insertions(
    customer: int,
    route_idx: int,
    source_route: Sequence[int],
    model: Any,
    base: RouteEvaluation | None = None,
) -> list[InsertionOption]:
    """Enumerate feasible positions in one route using a reusable base schedule."""
    route = list(source_route)
    if len(route) < 2:
        return []
    if base is None:
        base = evaluate_route(route, model)
    if not base.feasible:
        return []

    customer_demand = float(_node(model, customer).demand)
    if base.load + customer_demand > float(model.vehicle_cap) + FEASIBILITY_TOL:
        return []

    options: list[InsertionOption] = []
    for position in range(1, len(route)):
        trial = route[:position] + [customer] + route[position:]
        evaluated = evaluate_route(trial, model)
        if not evaluated.feasible:
            continue

        delays = [
            max(
                0.0,
                evaluated.service_starts[old_position + 1]
                - base.service_starts[old_position],
            )
            for old_position in range(position, len(route) - 1)
        ]
        options.append(InsertionOption(
            customer=customer,
            route_idx=route_idx,
            position=position,
            is_new_route=False,
            distance_delta=evaluated.distance - base.distance,
            waiting_delta=evaluated.total_waiting - base.total_waiting,
            total_downstream_delay=sum(delays),
            max_downstream_delay=max(delays, default=0.0),
            slack_loss=max(
                0.0,
                base.min_normalized_slack - evaluated.min_normalized_slack,
            ),
            min_slack_after=evaluated.min_normalized_slack,
        ))
    return options


def _singleton_insertion_option(
    customer: int,
    model: Any,
) -> InsertionOption | None:
    depot = _depot_id(model)
    evaluated = evaluate_route([depot, customer, depot], model)
    if not evaluated.feasible:
        return None
    return InsertionOption(
        customer=customer,
        route_idx=-1,
        position=1,
        is_new_route=True,
        distance_delta=evaluated.distance,
        waiting_delta=evaluated.total_waiting,
        total_downstream_delay=0.0,
        max_downstream_delay=0.0,
        slack_loss=0.0,
        min_slack_after=evaluated.min_normalized_slack,
    )


def _build_insertion_cache(
    customers: Sequence[int],
    routes: Sequence[Sequence[int]],
    model: Any,
) -> dict[int, dict[int, list[InsertionOption]]]:
    """Cache candidates by customer and route for incremental repair updates."""
    cache = {int(customer): {} for customer in customers}
    base_evaluations = [evaluate_route(route, model) for route in routes]

    for route_idx, (route, base) in enumerate(zip(routes, base_evaluations)):
        for customer in customers:
            options = _enumerate_route_insertions(
                customer,
                route_idx,
                route,
                model,
                base,
            )
            if options:
                cache[int(customer)][route_idx] = options

    for customer in customers:
        singleton = _singleton_insertion_option(customer, model)
        if singleton is not None:
            cache[int(customer)][-1] = [singleton]
    return cache


def _refresh_cached_route(
    cache: dict[int, dict[int, list[InsertionOption]]],
    customers: Sequence[int],
    route_idx: int,
    route: Sequence[int],
    model: Any,
) -> None:
    """Refresh only the route changed by the previous insertion."""
    base = evaluate_route(route, model)
    for customer in customers:
        customer_cache = cache[int(customer)]
        options = _enumerate_route_insertions(
            customer,
            route_idx,
            route,
            model,
            base,
        )
        if options:
            customer_cache[route_idx] = options
        else:
            customer_cache.pop(route_idx, None)


def _cached_options(
    cache: dict[int, dict[int, list[InsertionOption]]],
    customer: int,
) -> list[InsertionOption]:
    return [
        option
        for route_options in cache[int(customer)].values()
        for option in route_options
    ]


def _apply_option(
    routes: list[list[int]],
    customer: int,
    option: InsertionOption,
    model: Any,
) -> None:
    if option.is_new_route:
        depot = _depot_id(model)
        routes.append([depot, customer, depot])
        return
    routes[option.route_idx].insert(option.position, customer)


def _best_by_route(options: Sequence[InsertionOption]) -> list[InsertionOption]:
    best: dict[int, InsertionOption] = {}
    for option in options:
        current = best.get(option.route_idx)
        key = (
            option.distance_delta,
            option.is_new_route,
            option.position,
        )
        if current is None or key < (
            current.distance_delta,
            current.is_new_route,
            current.position,
        ):
            best[option.route_idx] = option
    return sorted(
        best.values(),
        key=lambda option: (
            option.distance_delta,
            option.is_new_route,
            option.route_idx,
            option.position,
        ),
    )


def _solution_distance(routes: Sequence[Sequence[int]], model: Any) -> float:
    total = 0.0
    for route in routes:
        evaluated = evaluate_route(route, model)
        if not evaluated.feasible:
            return float("inf")
        total += evaluated.distance
    return total


def _solution_signature(routes: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(route[1:-1]) for route in routes if len(route) > 2))


def validate_solution(
    routes: Sequence[Sequence[int]],
    expected_customers: Iterable[int],
    model: Any,
) -> tuple[bool, str]:
    expected = sorted(int(customer) for customer in expected_customers)
    actual = sorted(
        int(customer)
        for route in routes
        for customer in route[1:-1]
    )
    if actual != expected:
        return False, "Customer coverage contains omissions or duplicates."
    for route in routes:
        if not evaluate_route(route, model).feasible:
            return False, f"Infeasible route: {list(route)}"
    return True, ""


# ---------------------------------------------------------------------------
# Destroy operators: random node, dynamic node, cross-route cluster,
# route-local string, and complete-route elimination.
# ---------------------------------------------------------------------------


def destroy_random(
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
) -> tuple[list[list[int]], list[int]]:
    rng = _rng_or_default(rng)
    all_customers = [customer for route in routes for customer in route[1:-1]]
    if k <= 0 or not all_customers:
        return _active_route_copies(routes), []
    removed = list(rng.sample(all_customers, min(k, len(all_customers))))
    removed_set = set(removed)
    partial = [
        [customer for customer in route if customer not in removed_set]
        for route in routes
    ]
    return _clean_routes(partial), removed


def _removal_metrics(
    route: Sequence[int],
    position: int,
    model: Any,
) -> tuple[float, float, float]:
    base = evaluate_route(route, model)
    customer = route[position]
    reduced = list(route[:position]) + list(route[position + 1:])
    reduced_eval = evaluate_route(reduced, model)
    if not base.feasible or not reduced_eval.feasible:
        return 0.0, 0.0, 0.0

    delay_release = 0.0
    for old_position in range(position + 1, len(route) - 1):
        delay_release += max(
            0.0,
            base.service_starts[old_position]
            - reduced_eval.service_starts[old_position - 1],
        )

    node = _node(model, customer)
    width = max(float(node.end_time - node.start_time), EPS)
    pressure = min(1.0, max(
        0.0,
        (base.service_starts[position] - float(node.start_time)) / width,
    ))
    return base.distance - reduced_eval.distance, delay_release, pressure


def destroy_tw_worst(
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
    randomization: float = 5.0,
) -> tuple[list[list[int]], list[int]]:
    rng = _rng_or_default(rng)
    working = _active_route_copies(routes)
    target = min(max(0, k), sum(len(route) - 2 for route in working))
    removed: list[int] = []

    while len(removed) < target:
        records = []
        for route_idx, route in enumerate(working):
            for position in range(1, len(route) - 1):
                distance_saving, delay_release, pressure = _removal_metrics(
                    route, position, model
                )
                records.append((
                    route[position],
                    route_idx,
                    position,
                    distance_saving,
                    delay_release,
                    pressure,
                ))
        if not records:
            break

        distance_ranks = _rank01([record[3] for record in records])
        delay_ranks = _rank01([record[4] for record in records])
        pressure_ranks = _rank01([record[5] for record in records])
        ranked = []
        for record, distance_rank, delay_rank, pressure_rank in zip(
            records, distance_ranks, delay_ranks, pressure_ranks
        ):
            score = (
                0.50 * distance_rank
                + 0.30 * delay_rank
                + 0.20 * pressure_rank
            )
            ranked.append((score, record[0], record[1], record[2]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        _, customer, route_idx, position = _biased_pick(
            ranked, rng, randomization
        )
        removed.append(customer)
        del working[route_idx][position]
        if len(working[route_idx]) <= 2:
            del working[route_idx]

    return working, removed


def _matrix_max(matrix: Any) -> float:
    try:
        return max(float(matrix.max()), EPS)
    except AttributeError:
        return max(
            max(float(value) for value in row)
            for row in matrix
        )


def destroy_cross_route_shaw_tw(
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
    randomization: float = 6.0,
) -> tuple[list[list[int]], list[int]]:
    rng = _rng_or_default(rng)
    all_customers = [customer for route in routes for customer in route[1:-1]]
    if k <= 0 or not all_customers:
        return _active_route_copies(routes), []
    target = min(k, len(all_customers))

    route_of = {}
    service_start = {}
    for route_idx, route in enumerate(routes):
        evaluated = evaluate_route(route, model)
        for position in range(1, len(route) - 1):
            customer = route[position]
            route_of[customer] = route_idx
            if evaluated.feasible:
                service_start[customer] = evaluated.service_starts[position]

    max_distance = _matrix_max(model.distance_matrix)
    max_demand = max(
        max(float(_node(model, customer).demand) for customer in all_customers),
        EPS,
    )
    horizon = max(
        float(model.depot_end_time - model.depot_start_time), EPS
    )

    removed = [int(rng.choice(all_customers))]
    while len(removed) < target:
        reference = int(rng.choice(removed))
        candidates = [
            customer for customer in all_customers if customer not in removed
        ]
        cross_route = [
            customer
            for customer in candidates
            if route_of.get(customer) != route_of.get(reference)
        ]
        if cross_route:
            candidates = cross_route
        if not candidates:
            break

        reference_node = _node(model, reference)
        reference_midpoint = 0.5 * (
            float(reference_node.start_time) + float(reference_node.end_time)
        )
        scored = []
        for customer in candidates:
            node = _node(model, customer)
            midpoint = 0.5 * (
                float(node.start_time) + float(node.end_time)
            )
            window_gap = max(
                0.0,
                max(float(reference_node.start_time), float(node.start_time))
                - min(float(reference_node.end_time), float(node.end_time)),
            )
            relatedness = (
                0.35 * _distance(model, reference, customer) / max_distance
                + 0.15 * abs(
                    float(reference_node.demand) - float(node.demand)
                ) / max_demand
                + 0.25 * abs(
                    service_start.get(reference, reference_midpoint)
                    - service_start.get(customer, midpoint)
                ) / horizon
                + 0.15 * abs(reference_midpoint - midpoint) / horizon
                + 0.10 * window_gap / horizon
            )
            scored.append((relatedness, customer))
        scored.sort(key=lambda item: (item[0], item[1]))
        _, selected = _biased_pick(scored, rng, randomization)
        removed.append(selected)

    removed_set = set(removed)
    partial = [
        [customer for customer in route if customer not in removed_set]
        for route in routes
    ]
    return _clean_routes(partial), removed


def _block_metrics(
    route: Sequence[int],
    first_position: int,
    block_length: int,
    model: Any,
) -> tuple[float, float, float]:
    base = evaluate_route(route, model)
    last_position = first_position + block_length
    reduced = list(route[:first_position]) + list(route[last_position:])
    reduced_eval = evaluate_route(reduced, model)
    if not base.feasible or not reduced_eval.feasible:
        return 0.0, 0.0, 0.0

    time_relief = 0.0
    for old_position in range(last_position, len(route) - 1):
        time_relief += max(
            0.0,
            base.service_starts[old_position]
            - reduced_eval.service_starts[old_position - block_length],
        )
    slack_improvement = max(
        0.0,
        reduced_eval.min_normalized_slack - base.min_normalized_slack,
    )
    distance_saving = base.distance - reduced_eval.distance
    return time_relief, distance_saving, slack_improvement


def destroy_sisr_string(
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
    randomization: float = 4.0,
) -> tuple[list[list[int]], list[int]]:
    rng = _rng_or_default(rng)
    working = _active_route_copies(routes)
    target = min(max(0, k), sum(len(route) - 2 for route in working))
    removed: list[int] = []
    maximum_string = max(2, int(math.ceil(math.sqrt(max(target, 1)))))

    while len(removed) < target:
        remaining = target - len(removed)
        records = []
        for route_idx, route in enumerate(working):
            customer_count = len(route) - 2
            max_length = min(customer_count, remaining, maximum_string)
            if max_length <= 0:
                continue
            min_length = 2 if max_length >= 2 else 1
            for block_length in range(min_length, max_length + 1):
                for first_position in range(
                    1, len(route) - block_length
                ):
                    metrics = _block_metrics(
                        route, first_position, block_length, model
                    )
                    records.append((
                        route_idx,
                        first_position,
                        block_length,
                        *metrics,
                    ))
        if not records:
            break

        relief_ranks = _rank01([record[3] for record in records])
        distance_ranks = _rank01([record[4] for record in records])
        slack_ranks = _rank01([record[5] for record in records])
        ranked = []
        for record, relief_rank, distance_rank, slack_rank in zip(
            records, relief_ranks, distance_ranks, slack_ranks
        ):
            score = (
                0.60 * relief_rank
                + 0.25 * distance_rank
                + 0.15 * slack_rank
            )
            ranked.append((score, record[0], record[1], record[2]))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, route_idx, first_position, block_length = _biased_pick(
            ranked, rng, randomization
        )
        block = working[route_idx][
            first_position:first_position + block_length
        ]
        removed.extend(block)
        del working[route_idx][
            first_position:first_position + block_length
        ]
        if len(working[route_idx]) <= 2:
            del working[route_idx]

    return working, removed


def destroy_whole_route(
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
) -> tuple[list[list[int]], list[int]]:
    rng = _rng_or_default(rng)
    working = _active_route_copies(routes)
    target = min(max(0, k), sum(len(route) - 2 for route in working))
    if target <= 0 or not working:
        return working, []

    evaluations = [evaluate_route(route, model) for route in working]
    best_merge_saving = [0.0 for _ in working]
    merge_feasible = [False for _ in working]
    for first_idx, first in enumerate(working):
        first_customers = list(first[1:-1])
        for second_idx in range(first_idx + 1, len(working)):
            second = working[second_idx]
            second_customers = list(second[1:-1])
            base_distance = (
                evaluations[first_idx].distance
                + evaluations[second_idx].distance
            )
            for merged_customers in (
                first_customers + second_customers,
                second_customers + first_customers,
            ):
                merged = [_depot_id(model)] + merged_customers + [_depot_id(model)]
                merged_eval = evaluate_route(merged, model)
                if not merged_eval.feasible:
                    continue
                saving = max(0.0, base_distance - merged_eval.distance)
                merge_feasible[first_idx] = True
                merge_feasible[second_idx] = True
                best_merge_saving[first_idx] = max(
                    best_merge_saving[first_idx], saving
                )
                best_merge_saving[second_idx] = max(
                    best_merge_saving[second_idx], saving
                )

    records = []
    for route_idx, route in enumerate(working):
        evaluated = evaluations[route_idx]
        customer_count = len(route) - 2
        load_ratio = evaluated.load / max(float(model.vehicle_cap), EPS)
        roundtrip_sum = sum(
            _distance(model, _depot_id(model), customer)
            + _distance(model, customer, _depot_id(model))
            for customer in route[1:-1]
        )
        detour_ratio = evaluated.distance / max(roundtrip_sum, EPS)
        records.append((
            route_idx,
            customer_count,
            1.0 / max(customer_count, 1),
            1.0 - min(1.0, load_ratio),
            detour_ratio,
            float(merge_feasible[route_idx]),
            best_merge_saving[route_idx],
        ))

    length_ranks = _rank01([record[2] for record in records])
    load_ranks = _rank01([record[3] for record in records])
    detour_ranks = _rank01([record[4] for record in records])
    merge_saving_ranks = _rank01([record[6] for record in records])
    scored_routes = []
    for record, length_rank, load_rank, detour_rank, merge_saving_rank in zip(
        records,
        length_ranks,
        load_ranks,
        detour_ranks,
        merge_saving_ranks,
    ):
        score = (
            0.20 * length_rank
            + 0.15 * load_rank
            + 0.15 * detour_rank
            + 0.35 * record[5]
            + 0.15 * merge_saving_rank
            + 0.05 * rng.random()
        )
        scored_routes.append((record[0], record[1], score))

    # Route elimination is a focused move.  Matching a 10-25% global removal
    # target used to delete as many as five short routes at once, which removed
    # most of the destinations that their customers could use.  Search the best
    # one- or two-route subset instead; two routes still provide enough space for
    # cross-route consolidation without emptying a large part of the solution.
    max_eliminated_routes = 2
    subsets: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for route_idx, customer_count, score in scored_routes:
        updated = dict(subsets)
        for count, (total_score, indices) in subsets.items():
            if len(indices) >= max_eliminated_routes:
                continue
            new_count = count + customer_count
            candidate = (total_score + score, indices + (route_idx,))
            current = updated.get(new_count)
            if current is None or candidate[0] > current[0] + EPS:
                updated[new_count] = candidate
        subsets = updated

    _, (_, selected_indices) = min(
        ((count, value) for count, value in subsets.items() if count > 0),
        key=lambda item: (
            abs(item[0] - target),
            item[0] < target,
            -item[1][0],
            len(item[1][1]),
            item[1][1],
        ),
    )
    selected = set(selected_indices)
    removed = [
        customer
        for route_idx, route in enumerate(working)
        if route_idx in selected
        for customer in route[1:-1]
    ]
    partial = [
        route for route_idx, route in enumerate(working)
        if route_idx not in selected
    ]
    return partial, removed


# ---------------------------------------------------------------------------
# Repair operators: random greedy, true regret, fail-first criticality,
# time-impact minimization, and fast local one-customer ejection.
# ---------------------------------------------------------------------------


def _require_options(
    customer: int,
    routes: Sequence[Sequence[int]],
    model: Any,
) -> list[InsertionOption]:
    options = enumerate_feasible_insertions(customer, routes, model)
    if not options:
        original_id = getattr(_node(model, customer), "original_id", customer)
        raise ValueError(
            f"Customer {original_id} has no feasible insertion, including a new route."
        )
    return options


def repair_random_greedy(
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
    rcl_size: int = 3,
) -> list[list[int]]:
    """Insert in random customer order from a restricted candidate list.

    This is the deliberately exploratory repair direction.  Restricting the
    draw to a few low-cost positions prevents it from becoming pure random
    insertion, while drawing instead of always taking rank zero prevents it
    from collapsing into Regret-2 or deterministic greedy insertion.
    """
    rng = _rng_or_default(rng)
    working = _active_route_copies(routes)
    remaining = _unique_customers(removed, _depot_id(model))
    rng.shuffle(remaining)
    for customer in remaining:
        options = _require_options(customer, working, model)
        existing = [option for option in options if not option.is_new_route]
        if existing:
            options = existing
        options.sort(key=lambda option: (
            option.distance_delta,
            option.is_new_route,
            option.route_idx,
            option.position,
        ))
        candidate_count = min(max(1, int(rcl_size)), len(options))
        selected = options[rng.randrange(candidate_count)]
        _apply_option(working, customer, selected, model)
    return _clean_routes(working)


def repair_true_regret2(
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
) -> list[list[int]]:
    del rng
    working = _active_route_copies(routes)
    remaining = _unique_customers(removed, _depot_id(model))
    cache = _build_insertion_cache(remaining, working, model)

    while remaining:
        candidates = []
        for customer in remaining:
            options = _cached_options(cache, customer)
            if not options:
                _require_options(customer, working, model)
            route_options = _best_by_route(options)
            best = route_options[0]
            single_route_only = len(route_options) == 1
            regret = (
                route_options[1].distance_delta - best.distance_delta
                if not single_route_only else 0.0
            )
            candidates.append((
                not single_route_only,
                -regret,
                best.distance_delta,
                customer,
                best,
            ))
        _, _, _, customer, option = min(candidates)
        _apply_option(working, customer, option, model)
        remaining.remove(customer)
        cache.pop(customer, None)
        changed_route_idx = (
            len(working) - 1 if option.is_new_route else option.route_idx
        )
        _refresh_cached_route(
            cache,
            remaining,
            changed_route_idx,
            working[changed_route_idx],
            model,
        )

    return _clean_routes(working)


def _direct_normalized_slack(customer: int, model: Any) -> float:
    depot = _depot_id(model)
    evaluated = evaluate_route([depot, customer, depot], model)
    return evaluated.min_normalized_slack if evaluated.feasible else -1.0


def repair_most_constrained_first(
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
) -> list[list[int]]:
    rng = _rng_or_default(rng)
    working = _active_route_copies(routes)
    remaining = _unique_customers(removed, _depot_id(model))
    cache = _build_insertion_cache(remaining, working, model)

    while remaining:
        customer_records = []
        for customer in remaining:
            options = _cached_options(cache, customer)
            if not options:
                _require_options(customer, working, model)
            existing = [option for option in options if not option.is_new_route]
            route_count = len({option.route_idx for option in existing})
            node = _node(model, customer)
            customer_records.append((
                route_count,
                len(existing),
                _direct_normalized_slack(customer, model),
                float(node.end_time),
                customer,
                options,
            ))
        _, _, _, _, customer, options = min(customer_records)
        existing = [option for option in options if not option.is_new_route]
        candidate_options = existing if existing else options
        ranked_options = sorted(candidate_options, key=lambda option: (
            option.slack_loss,
            option.total_downstream_delay,
            option.max_downstream_delay,
            -option.min_slack_after,
            option.distance_delta,
            option.route_idx,
            option.position,
        ))
        best = ranked_options[rng.randrange(min(2, len(ranked_options)))]
        _apply_option(working, customer, best, model)
        remaining.remove(customer)
        cache.pop(customer, None)
        changed_route_idx = (
            len(working) - 1 if best.is_new_route else best.route_idx
        )
        _refresh_cached_route(
            cache,
            remaining,
            changed_route_idx,
            working[changed_route_idx],
            model,
        )

    return _clean_routes(working)


def repair_time_impact(
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
) -> list[list[int]]:
    del rng
    working = _active_route_copies(routes)
    remaining = _unique_customers(removed, _depot_id(model))

    while remaining:
        customer = min(
            remaining,
            key=lambda item: (
                _direct_normalized_slack(item, model),
                float(_node(model, item).end_time),
                float(_node(model, item).start_time),
                item,
            ),
        )
        options = _require_options(customer, working, model)
        existing = [option for option in options if not option.is_new_route]
        if existing:
            options = existing

        distance_ranks = _rank01([
            option.distance_delta for option in options
        ])
        waiting_ranks = _rank01([
            option.waiting_delta for option in options
        ])
        total_delay_ranks = _rank01([
            option.total_downstream_delay for option in options
        ])
        max_delay_ranks = _rank01([
            option.max_downstream_delay for option in options
        ])
        slack_ranks = _rank01([
            option.slack_loss for option in options
        ])

        scored = []
        for (
            option,
            distance_rank,
            waiting_rank,
            total_delay_rank,
            max_delay_rank,
            slack_rank,
        ) in zip(
            options,
            distance_ranks,
            waiting_ranks,
            total_delay_ranks,
            max_delay_ranks,
            slack_ranks,
        ):
            score = (
                0.10 * distance_rank
                + 0.25 * total_delay_rank
                + 0.25 * max_delay_rank
                + 0.20 * slack_rank
                + 0.15 * waiting_rank
                + 0.05 * float(option.is_new_route)
            )
            scored.append((
                score,
                option.distance_delta,
                option.is_new_route,
                option.route_idx,
                option.position,
                option,
            ))
        _, _, _, _, _, option = min(scored)
        _apply_option(working, customer, option, model)
        remaining.remove(customer)

    return _clean_routes(working)


def _route_compatibility(
    customer: int,
    route: Sequence[int],
    model: Any,
) -> float:
    route_customers = route[1:-1]
    if not route_customers:
        return 0.0
    max_distance = _matrix_max(model.distance_matrix)
    horizon = max(
        float(model.depot_end_time - model.depot_start_time), EPS
    )
    target_node = _node(model, customer)
    target_midpoint = 0.5 * (
        float(target_node.start_time) + float(target_node.end_time)
    )
    return min(
        0.65 * _distance(model, customer, other) / max_distance
        + 0.35 * abs(
            target_midpoint
            - 0.5 * (
                float(_node(model, other).start_time)
                + float(_node(model, other).end_time)
            )
        ) / horizon
        for other in route_customers
    )


def _ejection_positions(
    route: Sequence[int],
    target: int,
    model: Any,
    limit: int,
    excluded: Iterable[int] = (),
) -> list[int]:
    evaluated = evaluate_route(route, model)
    max_distance = _matrix_max(model.distance_matrix)
    excluded_set = set(excluded)
    target_demand = max(float(_node(model, target).demand), EPS)
    scored = []
    for position in range(1, len(route) - 1):
        customer = route[position]
        if customer in excluded_set:
            continue
        node = _node(model, customer)
        width = max(float(node.end_time - node.start_time), EPS)
        pressure = (
            evaluated.service_starts[position] - float(node.start_time)
        ) / width
        capacity_relief = min(1.0, float(node.demand) / target_demand)
        proximity = 1.0 - min(
            1.0, _distance(model, target, customer) / max_distance
        )
        score = (
            0.45 * capacity_relief
            + 0.35 * pressure
            + 0.20 * proximity
        )
        scored.append((-score, position))
    scored.sort()
    return [position for _, position in scored[:limit]]


def _nearby_insertion_positions(
    route: Sequence[int],
    removed_position: int,
    radius: int,
) -> list[int]:
    first = max(1, removed_position - max(0, radius))
    last = min(len(route) - 1, removed_position + max(0, radius))
    return list(range(first, last + 1))


def _fast_local_ejection_candidates(
    target: int,
    routes: Sequence[Sequence[int]],
    model: Any,
    excluded: Iterable[int],
    max_routes: int = 2,
    max_ejections_per_route: int = 2,
    position_radius: int = 1,
) -> list[tuple[tuple[float, ...], int, int, list[int]]]:
    """Rank a small set of local swaps before checking global reinsertion."""
    route_indices = sorted(
        (
            route_idx
            for route_idx, route in enumerate(routes)
            if len(route) > 3
        ),
        key=lambda route_idx: (
            _route_compatibility(target, routes[route_idx], model),
            route_idx,
        ),
    )[:max(0, int(max_routes))]

    candidates: list[tuple[tuple[float, ...], int, int, list[int]]] = []
    for route_idx in route_indices:
        route = list(routes[route_idx])
        base = evaluate_route(route, model)
        for eject_position in _ejection_positions(
            route,
            target,
            model,
            max(0, int(max_ejections_per_route)),
            excluded,
        ):
            ejected = route[eject_position]
            reduced = route[:eject_position] + route[eject_position + 1:]
            for insert_position in _nearby_insertion_positions(
                reduced, eject_position, position_radius
            ):
                trial = (
                    reduced[:insert_position]
                    + [target]
                    + reduced[insert_position:]
                )
                evaluated = evaluate_route(trial, model)
                if not evaluated.feasible:
                    continue
                key = (
                    evaluated.distance - base.distance,
                    -evaluated.min_normalized_slack,
                    route_idx,
                    eject_position,
                    insert_position,
                    ejected,
                )
                candidates.append((key, route_idx, ejected, trial))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _packing_priority(customer: int, model: Any) -> tuple[float, ...]:
    depot = _depot_id(model)
    node = _node(model, customer)
    singleton_distance = (
        _distance(model, depot, customer)
        + _distance(model, customer, depot)
    )
    return (
        -float(node.demand),
        -singleton_distance,
        float(node.end_time - node.start_time),
        float(node.end_time),
        customer,
    )


def _best_packing_option(
    customer: int,
    routes: Sequence[Sequence[int]],
    options: Sequence[InsertionOption],
    model: Any,
) -> InsertionOption | None:
    existing = [option for option in options if not option.is_new_route]
    if not existing:
        return None
    route_loads = {
        route_idx: evaluate_route(routes[route_idx], model).load
        for route_idx in {item.route_idx for item in existing}
    }
    customer_demand = float(_node(model, customer).demand)
    capacity_slacks = [
        float(model.vehicle_cap)
        - route_loads[item.route_idx]
        - customer_demand
        for item in existing
    ]
    distance_ranks = _rank01([
        item.distance_delta for item in existing
    ])
    capacity_ranks = _rank01(capacity_slacks)
    records = zip(existing, distance_ranks, capacity_ranks, capacity_slacks)
    return min(records, key=lambda record: (
        0.80 * record[1] + 0.20 * record[2],
        record[0].distance_delta,
        record[3],
        -record[0].min_slack_after,
        record[0].route_idx,
        record[0].position,
    ))[0]


def repair_fast_local_ejection(
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
    max_routes: int = 2,
    max_ejections_per_route: int = 2,
    position_radius: int = 1,
    max_total_ejections: int = 4,
    max_reinsertion_checks: int = 2,
) -> list[list[int]]:
    """Pack routes using bounded one-customer ejection and fleet fallback.

    Only the best few local swaps are checked.  A swap is accepted only when the
    ejected incumbent can immediately enter another existing route, so ejection
    never creates an extra route.  Unlimited fleet remains the feasibility fallback.
    """
    del rng
    working = _active_route_copies(routes)
    remaining = _unique_customers(removed, _depot_id(model))
    ejected_once: set[int] = set()
    protected: set[int] = set()
    ejection_count = 0
    ejection_limit = max(0, int(max_total_ejections))

    while remaining:
        target = min(remaining, key=lambda item: _packing_priority(item, model))
        options = _require_options(target, working, model)
        option = _best_packing_option(target, working, options, model)

        if option is not None:
            _apply_option(working, target, option, model)
        else:
            best_swap_key: tuple[float, ...] | None = None
            best_swap_routes: list[list[int]] | None = None
            best_ejected: int | None = None
            if ejection_count < ejection_limit and target not in ejected_once:
                candidates = _fast_local_ejection_candidates(
                    target=target,
                    routes=working,
                    model=model,
                    excluded=protected | ejected_once,
                    max_routes=max_routes,
                    max_ejections_per_route=max_ejections_per_route,
                    position_radius=position_radius,
                )
                for (
                    candidate_key,
                    route_idx,
                    ejected,
                    replacement_route,
                ) in candidates[:max(0, int(max_reinsertion_checks))]:
                    candidate_routes = [list(route) for route in working]
                    candidate_routes[route_idx] = replacement_route
                    ejected_options = enumerate_feasible_insertions(
                        ejected, candidate_routes, model
                    )
                    ejected_option = _best_packing_option(
                        ejected, candidate_routes, ejected_options, model
                    )
                    if ejected_option is None:
                        continue
                    _apply_option(
                        candidate_routes, ejected, ejected_option, model
                    )
                    swap_key = (
                        candidate_key[0] + ejected_option.distance_delta,
                        candidate_key[1],
                        route_idx,
                        ejected,
                        ejected_option.route_idx,
                        ejected_option.position,
                    )
                    if best_swap_key is None or swap_key < best_swap_key:
                        best_swap_key = swap_key
                        best_swap_routes = candidate_routes
                        best_ejected = ejected

            if best_swap_routes is not None and best_ejected is not None:
                working = best_swap_routes
                ejected_once.add(best_ejected)
                protected.add(best_ejected)
                ejection_count += 1
            else:
                option = min(options, key=lambda item: (
                    item.is_new_route,
                    item.distance_delta,
                    item.route_idx,
                    item.position,
                ))
                _apply_option(working, target, option, model)

        protected.add(target)
        remaining.remove(target)

    return _clean_routes(working)


# Preserve the previous direct-call name while R5 now uses the fast algorithm.
repair_bounded_ejection = repair_fast_local_ejection


DESTROY_OPERATORS = (
    destroy_random,
    destroy_tw_worst,
    destroy_cross_route_shaw_tw,
    destroy_sisr_string,
    destroy_whole_route,
)

REPAIR_OPERATORS = (
    repair_random_greedy,
    repair_true_regret2,
    repair_most_constrained_first,
    repair_time_impact,
    repair_fast_local_ejection,
)

DESTROY_OPERATOR_NAMES = (
    "random_node",
    "dynamic_tw_worst",
    "cross_route_shaw_tw",
    "sisr_string",
    "merge_aware_route_elimination",
)

REPAIR_OPERATOR_NAMES = (
    "randomized_rcl_greedy",
    "cached_true_regret2",
    "scarcity_first_slack_preserving",
    "urgent_time_impact",
    "fast_local_ejection",
)


def apply_destroy_operator(
    operator_id: int,
    routes: Sequence[Sequence[int]],
    k: int,
    model: Any,
    rng: Any | None = None,
) -> tuple[list[list[int]], list[int]]:
    return DESTROY_OPERATORS[operator_id](routes, k, model, rng)


def apply_repair_operator(
    operator_id: int,
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    model: Any,
    rng: Any | None = None,
) -> list[list[int]]:
    return REPAIR_OPERATORS[operator_id](routes, removed, model, rng)


def _dac_destroy_adapter(
    operator_id: int,
    routes: Sequence[Sequence[int]],
    k: int,
    dist_mat: Any | None = None,
    demands: Any | None = None,
    capacity: float | None = None,
    model: Any | None = None,
    similarity_matrix: Any | None = None,
    rng: Any | None = None,
    **kwargs: Any,
) -> tuple[list[list[int]], list[int]]:
    """Accept the existing DAC destroy-call contract without changing DAC."""
    del dist_mat, demands, capacity, similarity_matrix, kwargs
    if model is None:
        raise ValueError("The DAC destroy adapter requires model=Model(...).")
    return apply_destroy_operator(operator_id, routes, k, model, rng)


def _dac_repair_adapter(
    operator_id: int,
    routes: Sequence[Sequence[int]],
    removed: Sequence[int],
    dist_mat: Any | None = None,
    demands: Any | None = None,
    capacity: float | None = None,
    model: Any | None = None,
    rng: Any | None = None,
    **kwargs: Any,
) -> list[list[int]]:
    """Accept the existing DAC repair-call contract without changing DAC."""
    del dist_mat, demands, capacity, kwargs
    if model is None:
        raise ValueError("The DAC repair adapter requires model=Model(...).")
    return apply_repair_operator(operator_id, routes, removed, model, rng)


# These registries are drop-in replacements for the embedded DESTROY_FUNCS and
# REPAIR_FUNCS used by DAC_ALNS_VRPTW_Online.py.
DAC_DESTROY_FUNCS = tuple(
    partial(_dac_destroy_adapter, operator_id)
    for operator_id in range(len(DESTROY_OPERATORS))
)
DAC_REPAIR_FUNCS = tuple(
    partial(_dac_repair_adapter, operator_id)
    for operator_id in range(len(REPAIR_OPERATORS))
)


__all__ = [
    "DESTROY_OPERATORS",
    "REPAIR_OPERATORS",
    "DAC_DESTROY_FUNCS",
    "DAC_REPAIR_FUNCS",
    "DESTROY_OPERATOR_NAMES",
    "REPAIR_OPERATOR_NAMES",
    "InsertionOption",
    "RouteEvaluation",
    "apply_destroy_operator",
    "apply_repair_operator",
    "enumerate_feasible_insertions",
    "evaluate_route",
    "validate_solution",
]
