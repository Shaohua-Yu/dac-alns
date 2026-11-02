# -*- coding: utf-8 -*-
# Single-AC-ALNS-VRPTW 在线控制变量版
# 单 Actor 使用23维节点特征一次选择25个破坏-修复算子对，在每个 VRPTW 算例内在线更新。

import os
import csv
import math
import time
import json
import copy
import random
import argparse
from pathlib import Path

from vrptw_operators import (
    DAC_DESTROY_FUNCS as DESTROY_FUNCS,
    DAC_REPAIR_FUNCS as REPAIR_FUNCS,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

import matplotlib
matplotlib.use("Agg")  # 兼容无显示环境
import matplotlib.pyplot as plt

torch.set_num_threads(1)  # 限制 PyTorch 在 CPU 上使用的线程数

# ---------------------- 设备/随机种子 ---------------------- #
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NODE_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "demand_ratio",
    "depot_flag",
    "served_flag",
    "route_load_ratio",
    "route_position_ratio",
    "predecessor_distance",
    "successor_distance",
    "removal_saving",
    "route_distance_ratio",
    "route_customer_ratio",
    "ready_global",
    "due_global",
    "window_global",
    "service_local",
    "direct_wait_local",
    "direct_slack_local",
    "service_start_global",
    "current_wait_local",
    "local_time_pressure",
    "downstream_pressure",
    "route_finish_global",
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
FEATURE_DEPOT_FLAG = NODE_FEATURE_NAMES.index("depot_flag")
FEATURE_SERVED_FLAG = NODE_FEATURE_NAMES.index("served_flag")
FEATURE_READY_GLOBAL = NODE_FEATURE_NAMES.index("ready_global")
FEATURE_DIRECT_SLACK_LOCAL = NODE_FEATURE_NAMES.index("direct_slack_local")
FEATURE_CURRENT_WAIT_LOCAL = NODE_FEATURE_NAMES.index("current_wait_local")
FEATURE_LOCAL_TIME_PRESSURE = NODE_FEATURE_NAMES.index("local_time_pressure")
FEATURE_DOWNSTREAM_PRESSURE = NODE_FEATURE_NAMES.index("downstream_pressure")
GLOBAL_TIME_STATE_DIM = 5
CHECKPOINT_ARCHITECTURE = "single-ac-alns-vrptw-23d-v1"

# ---------------------- 结果追加与保存（文件锁并发安全版本） ---------------------- #
def safe_append_to_json(file_path, problem_name, obj, timestamp):
    file_path = str(file_path)
    lock_file = file_path + '.lock'
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_file) > 10:
                    os.remove(lock_file)
            except OSError:
                pass
            time.sleep(random.uniform(0.05, 0.2))
        except OSError:
            time.sleep(random.uniform(0.05, 0.2))
    try:
        data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except Exception:
                pass
        problem_data = data.setdefault(problem_name, {"objectives": [], "times": []})
        problem_data["objectives"].append(obj)
        problem_data["times"].append(timestamp)
        temp_file = file_path + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, file_path)
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass

def save_to_csv(file_path, record_dict):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=record_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record_dict)

# ---------------------- 基础数据结构 ---------------------- #
class Sol:
    def __init__(self):
        self.node_no_seq = None
        self.obj = None
        self.route_list = None
        self.route_distance = None
        self.timetable_list = None

class Node:
    def __init__(self):
        self.id = 0
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.demand = 0.0
        self.start_time = 0.0
        self.end_time = 1440.0
        self.service_time = 0.0
        self.original_id = 0

class Model:
    def __init__(self):
        self.best_sol = None
        self.node_id_list = []  # 不含depot
        self.demand_dict = {}  # id->Node（含depot）
        self.depot = None
        self.number_of_demands = 0
        self.vehicle_cap = 100
        self.vehicle_speed = 1.0
        self.depot_start_time = 0.0
        self.depot_end_time = 1440.0
        self.distance_matrix = {}
        self.time_matrix = {}
        self.distance_matrix_dict = {}
        self.demands_arr = None
        self.start_arr = None
        self.end_arr = None
        self.service_arr = None
        self.similarity_matrix = None  # 静态相似度矩阵

        # 算子控制
        self.rand_d_max = 0.25
        self.rand_d_min = 0.1
        self.worst_d_min = 5
        self.worst_d_max = 20
        self.regret_n = 2
        self.T0 = 50.0
        self.phi = 0.995

        # 统计量
        self.d_select = np.zeros(5)
        self.r_select = np.zeros(5)

# ---------------------- 读取/距离/DP 分裂 ---------------------- #
def readCsvFile(demand_file, model: Model):
    def value(row, names, default):
        for name in names:
            if name in row and str(row[name]).strip() != '':
                return float(row[name])
        return float(default)

    with open(demand_file, 'r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV 文件为空: {demand_file}")
    required = {'id', 'x_coord', 'y_coord'}
    if not required.issubset(rows[0].keys()):
        raise ValueError("CSV 缺少必要列: id, x_coord, y_coord")

    depot_candidates = []
    for row in rows:
        raw_id = int(float(row['id']))
        is_depot = str(row.get('is_depot', '')).strip().lower() in {'1', 'true', 'yes', 'y'}
        if is_depot or raw_id == 0:
            depot_candidates.append(row)
    if not depot_candidates:
        depot_candidates = [row for row in rows if value(row, ('demand',), 0.0) == 0.0]
    if len(depot_candidates) != 1:
        raise ValueError("CSV 必须且只能包含一个车场（id=0 或 is_depot=true）")

    depot_row = depot_candidates[0]
    customer_rows = [row for row in rows if row is not depot_row]
    ordered_rows = [depot_row] + customer_rows

    model.vehicle_cap = value(depot_row, ('v_cap', 'capacity'), value(rows[0], ('v_cap', 'capacity'), 100.0))
    model.vehicle_speed = max(value(depot_row, ('v_speed', 'speed'), 1.0), 1e-9)
    model.depot_start_time = value(depot_row, ('depot_start_time', 'ready_time', 'start_time'), 0.0)
    model.depot_end_time = value(
        depot_row, ('depot_end_time', 'due_time', 'due_date', 'end_time'), 1440.0)

    for internal_id, row in enumerate(ordered_rows):
        node = Node()
        node.id = internal_id
        node.original_id = int(float(row['id']))
        node.x_coord = float(row['x_coord'])
        node.y_coord = float(row['y_coord'])
        if internal_id == 0:
            node.demand = 0.0
            node.start_time = model.depot_start_time
            node.end_time = model.depot_end_time
            node.service_time = 0.0
            model.depot = node
        else:
            node.demand = value(row, ('demand',), 0.0)
            node.start_time = value(row, ('ready_time', 'start_time'), 0.0)
            node.end_time = value(
                row, ('due_time', 'due_date', 'end_time'), model.depot_end_time)
            node.service_time = value(row, ('service_time', 'service'), 0.0)
            model.node_id_list.append(node.id)
        model.demand_dict[node.id] = node

    model.number_of_demands = len(model.node_id_list)

def calDistanceMatrix(model):
    ids = [model.depot.id] + model.node_id_list
    coords = np.array([[model.demand_dict[i].x_coord,
                        model.demand_dict[i].y_coord] for i in ids],
                      dtype=np.float32)

    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(-1, dtype=np.float32))

    model.distance_matrix = dist_mat
    model.time_matrix = dist_mat / max(float(model.vehicle_speed), 1e-9)
    model.id2idx = {id_: idx for idx, id_ in enumerate(ids)}
    model.idx2id = ids

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            model.distance_matrix_dict[(id_i, id_j)] = dist_mat[i, j]

    # 初始化需求快速查询
    max_id = max(model.demand_dict.keys())
    model.demands_arr = np.zeros(max_id + 1, dtype=np.float32)
    model.start_arr = np.zeros(max_id + 1, dtype=np.float32)
    model.end_arr = np.zeros(max_id + 1, dtype=np.float32)
    model.service_arr = np.zeros(max_id + 1, dtype=np.float32)
    for nid, node in model.demand_dict.items():
        model.demands_arr[nid] = node.demand
        model.start_arr[nid] = node.start_time
        model.end_arr[nid] = node.end_time
        model.service_arr[nid] = node.service_time

    # 保留与 DAC 一致的静态 Shaw 相似度，当前 5x5 算子内部使用动态相关度。
    n = len(ids)
    sim_mat = np.zeros((n, n), dtype=np.float32)
    max_d = np.max(dist_mat) + 1e-9
    max_q = np.max(model.demands_arr) + 1e-9
    horizon = max(float(model.depot_end_time - model.depot_start_time), 1e-9)
    for u in range(n):
        for v in range(n):
            dist_score = dist_mat[u, v] / max_d
            demand_score = abs(model.demands_arr[u] - model.demands_arr[v]) / max_q
            midpoint_u = 0.5 * (model.start_arr[u] + model.end_arr[u])
            midpoint_v = 0.5 * (model.start_arr[v] + model.end_arr[v])
            midpoint_score = abs(midpoint_u - midpoint_v) / horizon
            intersection = max(
                0.0,
                min(model.end_arr[u], model.end_arr[v])
                - max(model.start_arr[u], model.start_arr[v])
            )
            union = (
                max(model.end_arr[u], model.end_arr[v])
                - min(model.start_arr[u], model.start_arr[v])
            )
            overlap_penalty = 1.0 - intersection / max(float(union), 1e-9)
            sim_mat[u, v] = (
                0.45 * dist_score
                + 0.15 * demand_score
                + 0.20 * midpoint_score
                + 0.20 * overlap_penalty
            )
    model.similarity_matrix = sim_mat

    infeasible = []
    for customer_id in model.node_id_list:
        route = [model.depot.id, customer_id, model.depot.id]
        feasible, _, _ = calRouteSchedule(route, model)
        if not feasible:
            infeasible.append(model.demand_dict[customer_id].original_id)
    if infeasible:
        raise ValueError(f"以下客户单独成路线仍不可行: {infeasible}")

def genInitialSol(model: Model):
    unvisited = copy.deepcopy(model.node_id_list)
    if not unvisited:
        return []
    seq = []
    curr = 0
    while unvisited:
        nearest = min(
            unvisited,
            key=lambda x: (
                model.demand_dict[x].end_time,
                model.distance_matrix[model.id2idx[curr], model.id2idx[x]]
            )
        )
        seq.append(nearest)
        unvisited.remove(nearest)
        curr = nearest
    return seq

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

def _split_dp_core_py(seq, dem_arr, start_arr, end_arr, service_arr,
                      Q, dist_mat, time_mat, depot_idx, depot_start, depot_end):
    n = seq.shape[0]
    F = np.full(n + 1, np.inf, dtype=np.float64)
    pred = np.full(n + 1, -1, dtype=np.int32)
    F[0] = 0.0

    for i in range(n):
        if not np.isfinite(F[i]):
            continue
        load = 0.0
        route_distance = 0.0
        clock = depot_start
        previous = depot_idx

        for j in range(i, n):
            current = seq[j]
            load += dem_arr[current]
            raw_arrival = clock + time_mat[previous, current]
            service_start = max(raw_arrival, start_arr[current])
            departure = service_start + service_arr[current]
            route_distance += dist_mat[previous, current]

            feasible = (
                load <= Q
                and service_start <= end_arr[current]
                and departure + time_mat[current, depot_idx] <= depot_end
            )
            if not feasible:
                break

            complete_cost = route_distance + dist_mat[current, depot_idx]
            if F[i] + complete_cost < F[j + 1]:
                F[j + 1] = F[i] + complete_cost
                pred[j + 1] = i

            clock = departure
            previous = current

    return pred, float(F[n])

if _NUMBA:
    _split_dp_core = njit(_split_dp_core_py, cache=True, fastmath=True)
else:
    _split_dp_core = _split_dp_core_py

def splitRoutes(node_no_seq, model):
    if not node_no_seq:
        return [], 0.0

    seq_idx = np.array([model.id2idx[id_] for id_ in node_no_seq], dtype=np.int32)
    dem_arr = np.array([model.demand_dict[id_].demand for id_ in node_no_seq],
                       dtype=np.float32)

    pred, total_cost = _split_dp_core(
        seq_idx,
        model.demands_arr,
        model.start_arr,
        model.end_arr,
        model.service_arr,
        model.vehicle_cap,
        model.distance_matrix,
        model.time_matrix,
        depot_idx=0,
        depot_start=model.depot_start_time,
        depot_end=model.depot_end_time
    )

    if not math.isfinite(total_cost) or int(pred[len(seq_idx)]) < 0:
        return [], float('inf')

    routes_idx = []
    i = len(seq_idx)
    while i > 0:
        j = int(pred[i])
        routes_idx.append(seq_idx[j:i])
        i = j

    routes_idx.reverse()

    depot_id = model.depot.id
    route_list = [
        [depot_id] + [model.idx2id[int(idx)] for idx in r] + [depot_id]
        for r in routes_idx
    ]
    return route_list, total_cost

def calRouteDistance(route, model):
    return sum(
        model.distance_matrix[
            model.id2idx[route[i]],
            model.id2idx[route[i + 1]]
        ]
        for i in range(len(route) - 1)
    )


def calRouteSchedule(route, model):
    if len(route) < 2 or route[0] != model.depot.id or route[-1] != model.depot.id:
        return False, float('inf'), []

    clock = float(model.depot_start_time)
    load = 0.0
    distance = 0.0
    timetable = [(clock, clock)]

    for previous_id, current_id in zip(route, route[1:]):
        previous_idx = model.id2idx[previous_id]
        current_idx = model.id2idx[current_id]
        distance += float(model.distance_matrix[previous_idx, current_idx])
        raw_arrival = clock + float(model.time_matrix[previous_idx, current_idx])

        if current_id == model.depot.id:
            if raw_arrival > model.depot_end_time + 1e-7:
                return False, float('inf'), []
            clock = raw_arrival
            timetable.append((clock, clock))
            continue

        node = model.demand_dict[current_id]
        load += node.demand
        service_start = max(raw_arrival, node.start_time)
        if load > model.vehicle_cap + 1e-7 or service_start > node.end_time + 1e-7:
            return False, float('inf'), []
        departure = service_start + node.service_time
        timetable.append((service_start, departure))
        clock = departure

    return True, distance, timetable


def calObj(node_no_seq, model):
    route_list, total_distance = splitRoutes(node_no_seq, model)
    if not route_list:
        return float('inf'), [], [], []
    route_distance = []
    timetable_list = []
    for route in route_list:
        feasible, distance, timetable = calRouteSchedule(route, model)
        if not feasible:
            return float('inf'), [], [], []
        route_distance.append(float(distance))
        timetable_list.append(timetable)

    return float(total_distance), route_list, route_distance, timetable_list


def apply_destroy_to_partial_sol_explicit(destroyed_routes, model: Model):
    """
    方案A：直接基于显式残破路线构建 Partial Sol，不调用 DP Split
    """
    part = Sol()
    part.route_list = destroyed_routes

    route_distances = []
    timetable_list = []
    for route in destroyed_routes:
        feasible, distance, timetable = calRouteSchedule(route, model)
        if not feasible:
            part.obj = float('inf')
            part.route_distance = []
            part.timetable_list = []
            part.node_no_seq = []
            return part
        route_distances.append(float(distance))
        timetable_list.append(timetable)

    part.route_distance = route_distances
    part.timetable_list = timetable_list
    part.obj = sum(route_distances)

    # 一维序列仅用作状态记录（如 GNN 输入），通过扁平化路线直接获得
    part.node_no_seq = [c for r in destroyed_routes for c in r[1:-1]]
    return part

# 5x5 operators are imported from vrptw_operators_5x5.py.
def routes_to_seq(routes):
    return [c for r in routes for c in r[1:-1]]

# ==================== 接口适配器 ==================== #
def doDestroy_by_id(did, model: Model, sol: Sol, it, num_steps):
    routes = sol.route_list if sol.route_list else [[0] + sol.node_no_seq + [0]]
    num_cust = len(model.node_id_list)
    if num_cust <= 1:
        return routes, []

    d_max_current = model.rand_d_min + (model.rand_d_max - model.rand_d_min) * math.exp(-3.0 * it / num_steps)
    d = random.uniform(model.rand_d_min, d_max_current)
    k = min(max(1, int(d * num_cust)), num_cust - 1 if num_cust > 1 else 0)

    if did == 2:
        new_routes, removed = DESTROY_FUNCS[did](
            routes, k, dist_mat=model.distance_matrix, demands=model.demands_arr,
            capacity=model.vehicle_cap, similarity_matrix=model.similarity_matrix,
            model=model,
        )
    else:
        new_routes, removed = DESTROY_FUNCS[did](
            routes, k, dist_mat=model.distance_matrix, demands=model.demands_arr,
            capacity=model.vehicle_cap, model=model,
        )
    return new_routes,removed


def doRepair_by_id(rid, destroyed_routes, remove_list, model: Model, sol=None, *args, **kwargs):
    """
    方案A：直接在残破路线上执行插入，并直接计算最终距离，杜绝 DP Split
    （兼容版本：支持接收 5 个或更多参数，自动忽略多余的 sol）
    """
    new_routes = REPAIR_FUNCS[rid](
        destroyed_routes, remove_list, dist_mat=model.distance_matrix,
        demands=model.demands_arr, capacity=model.vehicle_cap, model=model
    )

    new_sol = Sol()
    new_sol.route_list = [r for r in new_routes if len(r) > 2]

    route_distances = []
    timetable_list = []
    for route in new_sol.route_list:
        feasible, distance, timetable = calRouteSchedule(route, model)
        if not feasible:
            new_sol.obj = float('inf')
            new_sol.route_distance = []
            new_sol.timetable_list = []
            new_sol.node_no_seq = []
            return new_sol
        route_distances.append(float(distance))
        timetable_list.append(timetable)

    new_sol.route_distance = route_distances
    new_sol.timetable_list = timetable_list
    new_sol.obj = sum(route_distances)
    new_sol.node_no_seq = [c for r in new_sol.route_list for c in r[1:-1]]
    return new_sol

# ---------------------- 预计算 ---------------------- #
def precompute_base_graph(node_list, model: Model, k_nn=10, use_full_graph_knn=False, weight_mode='inv'):
    n = len(node_list)
    edges = set()

    coords = np.array([[node.x_coord, node.y_coord] for node in node_list], dtype=np.float32)
    coords_shifted = coords - coords.min(axis=0)
    coord_scale = max(
        float(coords_shifted[:, 0].max()),
        float(coords_shifted[:, 1].max()),
        1e-6
    )
    norm_coords = coords_shifted / coord_scale

    diff = norm_coords[:, None, :] - norm_coords[None, :, :]
    norm_dist_mat = np.sqrt((diff ** 2).sum(-1, dtype=np.float32))

    if use_full_graph_knn:
        for u in range(n):
            for v in range(n):
                if u != v:
                    edges.add((u, v))
    else:
        k = min(k_nn, max(1, n - 1))
        for u in range(n):
            dists = [(v, norm_dist_mat[u, v]) for v in range(n) if v != u]
            dists.sort(key=lambda x: x[1])
            for v, _ in dists[:k]:
                edges.add((u, v))
                edges.add((v, u))

    if not edges:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        ew = torch.empty((0,), dtype=torch.float, device=device)
        return ei, ew

    ei = torch.tensor(list(edges), dtype=torch.long, device=device).t().contiguous()

    if weight_mode == 'dist':
        ew = torch.tensor([float(norm_dist_mat[u, v]) for (u, v) in edges],
                          dtype=torch.float, device=device)
    else:
        ew = torch.tensor([1.0 / (1.0 + float(norm_dist_mat[u, v])) for (u, v) in edges],
                          dtype=torch.float, device=device)
    return ei, ew

# ---------------------- 构图：12维 CVRP 特征 + 6维静态时间窗特征 + 5维动态时间特征 ---------------------- #
def build_graph(
        sol: Sol,
        node_list,
        model: Model,
        base_edge_index,
        base_edge_weight,
        removed_set=None,
        include_path_edges=True
):
    n = len(node_list)
    removed_set = set() if removed_set is None else set(removed_set)

    coords = np.array(
        [
            [node.x_coord, node.y_coord]
            for node in node_list
        ],
        dtype=np.float32
    )

    coords_shifted = coords - coords.min(axis=0)

    coord_scale = max(
        float(coords_shifted[:, 0].max()),
        float(coords_shifted[:, 1].max()),
        1e-6
    )

    norm_coords = coords_shifted / coord_scale

    capacity = max(float(model.vehicle_cap), 1e-6)
    num_customers = max(1, len(model.node_id_list))
    max_distance = max(
        float(np.max(model.distance_matrix)),
        1e-6
    )

    instance_distance_scale = max(
        max_distance * num_customers,
        1e-6
    )
    depot_start = float(model.depot_start_time)
    horizon = max(float(model.depot_end_time - model.depot_start_time), 1e-6)

    served_flag = np.ones(n, dtype=np.float32)

    for idx, node in enumerate(node_list):
        if node.id in removed_set:
            served_flag[idx] = 0.0

    route_load_ratio = np.zeros(n, dtype=np.float32)
    route_position_ratio = np.zeros(n, dtype=np.float32)
    predecessor_distance = np.zeros(n, dtype=np.float32)
    successor_distance = np.zeros(n, dtype=np.float32)
    removal_saving = np.zeros(n, dtype=np.float32)
    route_distance_ratio = np.zeros(n, dtype=np.float32)
    route_customer_ratio = np.zeros(n, dtype=np.float32)
    # Dynamic features are zero for unrouted customers. served_flag tells the
    # encoder that these zeroes represent a post-destroy customer.
    service_start_global = np.zeros(n, dtype=np.float32)
    current_wait_local = np.zeros(n, dtype=np.float32)
    local_time_pressure = np.zeros(n, dtype=np.float32)
    downstream_pressure = np.zeros(n, dtype=np.float32)
    route_finish_global = np.zeros(n, dtype=np.float32)

    if sol.route_list:
        for route in sol.route_list:
            if len(route) <= 2:
                continue

            customers = route[1:-1]
            customer_count = len(customers)

            route_load = sum(
                model.demand_dict[c].demand
                for c in customers
            )

            route_indices = [
                model.id2idx[node_id]
                for node_id in route
            ]

            route_distance = float(
                model.distance_matrix[
                    route_indices[:-1],
                    route_indices[1:]
                ].sum()
            )

            route_schedule = []
            clock = depot_start
            route_return_time = clock
            for previous_id, current_id in zip(route, route[1:]):
                previous_idx = model.id2idx[previous_id]
                current_idx = model.id2idx[current_id]
                raw_arrival = clock + float(model.time_matrix[previous_idx, current_idx])

                if current_id == model.depot.id:
                    route_return_time = raw_arrival
                    break

                current_node = model.demand_dict[current_id]
                service_start = max(raw_arrival, float(current_node.start_time))
                waiting_time = max(0.0, service_start - raw_arrival)
                departure = service_start + float(current_node.service_time)
                window_width = max(
                    float(current_node.end_time - current_node.start_time),
                    1e-6
                )
                pressure = float(np.clip(
                    (service_start - float(current_node.start_time)) / window_width,
                    0.0,
                    1.0
                ))
                route_schedule.append(
                    (
                        current_id,
                        service_start,
                        waiting_time,
                        window_width,
                        pressure
                    )
                )
                clock = departure

            finish_ratio = float(np.clip(
                (route_return_time - depot_start) / horizon,
                0.0,
                1.0
            ))

            max_future_pressure = 0.0
            for (
                    customer_id,
                    service_start,
                    waiting_time,
                    window_width,
                    pressure
            ) in reversed(route_schedule):
                node_idx = model.id2idx[customer_id]
                max_future_pressure = max(max_future_pressure, pressure)
                service_start_global[node_idx] = np.clip(
                    (service_start - depot_start) / horizon,
                    0.0,
                    1.0
                )
                current_wait_local[node_idx] = (
                    waiting_time / (waiting_time + window_width)
                )
                local_time_pressure[node_idx] = pressure
                downstream_pressure[node_idx] = max_future_pressure
                route_finish_global[node_idx] = finish_ratio

            for position, customer_id in enumerate(
                    customers,
                    start=1
            ):
                node_idx = model.id2idx[customer_id]

                predecessor_id = route[position - 1]
                successor_id = route[position + 1]

                predecessor_idx = model.id2idx[predecessor_id]
                successor_idx = model.id2idx[successor_id]

                d_pre = float(
                    model.distance_matrix[
                        predecessor_idx,
                        node_idx
                    ]
                )

                d_succ = float(
                    model.distance_matrix[
                        node_idx,
                        successor_idx
                    ]
                )

                d_bypass = float(
                    model.distance_matrix[
                        predecessor_idx,
                        successor_idx
                    ]
                )

                saving = d_pre + d_succ - d_bypass

                route_load_ratio[node_idx] = (
                    route_load / capacity
                )

                if customer_count <= 1:
                    route_position_ratio[node_idx] = 0.5
                else:
                    route_position_ratio[node_idx] = (
                        (position - 1)
                        / float(customer_count - 1)
                    )

                predecessor_distance[node_idx] = (
                    d_pre / max_distance
                )

                successor_distance[node_idx] = (
                    d_succ / max_distance
                )

                removal_saving[node_idx] = (
                    saving / max_distance
                )

                route_distance_ratio[node_idx] = (
                    route_distance / instance_distance_scale
                )

                route_customer_ratio[node_idx] = (
                    customer_count / float(num_customers)
                )

    x_data = []
    depot_idx = model.id2idx[model.depot.id]

    for idx, node in enumerate(node_list):
        depot_flag = (
            1.0
            if node.id == model.depot.id
            else 0.0
        )

        demand_ratio = (
            float(node.demand) / capacity
        )

        earliest_arrival = (
            depot_start
            + float(model.time_matrix[depot_idx, idx])
        )
        window_width = max(
            float(node.end_time - node.start_time),
            1e-6
        )
        direct_start = max(earliest_arrival, float(node.start_time))
        direct_wait = max(0.0, direct_start - earliest_arrival)

        ready_ratio = np.clip(
            (float(node.start_time) - depot_start) / horizon,
            0.0,
            1.0
        )
        due_ratio = np.clip(
            (float(node.end_time) - depot_start) / horizon,
            0.0,
            1.0
        )
        window_width_ratio = np.clip(window_width / horizon, 0.0, 1.0)
        service_ratio = np.clip(
            float(node.service_time) / window_width,
            0.0,
            1.0
        )
        initial_wait_ratio = direct_wait / (direct_wait + window_width)
        initial_tightness = np.clip(
            (float(node.end_time) - direct_start) / window_width,
            -1.0,
            1.0
        )

        x_data.append([
            float(norm_coords[idx, 0]),
            float(norm_coords[idx, 1]),
            demand_ratio,
            depot_flag,
            float(served_flag[idx]),
            float(route_load_ratio[idx]),
            float(route_position_ratio[idx]),
            float(predecessor_distance[idx]),
            float(successor_distance[idx]),
            float(removal_saving[idx]),
            float(route_distance_ratio[idx]),
            float(route_customer_ratio[idx]),
            float(ready_ratio),
            float(due_ratio),
            float(window_width_ratio),
            float(service_ratio),
            float(initial_wait_ratio),
            float(initial_tightness),
            float(service_start_global[idx]),
            float(current_wait_local[idx]),
            float(local_time_pressure[idx]),
            float(downstream_pressure[idx]),
            float(route_finish_global[idx])
        ])

    x = torch.tensor(
        x_data,
        dtype=torch.float,
        device=device
    )
    if x.size(1) != NODE_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {NODE_FEATURE_DIM} node features, got {x.size(1)}."
        )
    if not torch.isfinite(x).all():
        raise RuntimeError("Node features contain NaN or infinite values.")

    edge_index = base_edge_index
    edge_weight = base_edge_weight

    if include_path_edges and sol.route_list:
        dynamic_edges = set()

        for route in sol.route_list:
            for idx in range(len(route) - 1):
                u_id = route[idx]
                v_id = route[idx + 1]

                u_idx = model.id2idx[u_id]
                v_idx = model.id2idx[v_id]

                if (
                        0 <= u_idx < n
                        and 0 <= v_idx < n
                ):
                    dynamic_edges.add((u_idx, v_idx))
                    dynamic_edges.add((v_idx, u_idx))

        if dynamic_edges:
            dynamic_edge_list = list(dynamic_edges)

            dynamic_edge_index = torch.tensor(
                dynamic_edge_list,
                dtype=torch.long,
                device=device
            ).t().contiguous()

            dynamic_edge_weights = []

            for u_idx, v_idx in dynamic_edge_list:
                normalized_distance = float(
                    np.linalg.norm(
                        norm_coords[u_idx]
                        - norm_coords[v_idx]
                    )
                )

                dynamic_edge_weights.append(
                    1.0 / (1.0 + normalized_distance)
                )

            dynamic_edge_weight = torch.tensor(
                dynamic_edge_weights,
                dtype=torch.float,
                device=device
            )

            edge_index = torch.cat(
                [
                    edge_index,
                    dynamic_edge_index
                ],
                dim=1
            )

            edge_weight = torch.cat(
                [
                    edge_weight,
                    dynamic_edge_weight
                ],
                dim=0
            )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

# ---------------------- 缓冲区 ---------------------- #
class RolloutBuffer:
    """
    短期 on-policy rollout（单阶段联合动作消融版本）。
    每条记录包含单阶段马尔可夫决策轨迹（s_t, a_joint, r, s_t+1, done）。
    """
    def __init__(self):
        self.records = []

    @staticmethod
    def clone_state_to_cpu(state_record):
        (
            graph,
            stage,
            destroy_ratio,
            destroy_id,
            progress,
            temperature_ratio,
            best_gap,
            stagnation_ratio
        ) = state_record

        return (
            graph.clone().cpu(),
            int(stage),
            float(destroy_ratio),
            int(destroy_id),
            float(progress),
            float(temperature_ratio),
            float(best_gap),
            float(stagnation_ratio)
        )

    def add(
            self,
            state,
            joint_action,
            reward,
            next_state,
            done
    ):
        self.records.append({
            "state": self.clone_state_to_cpu(state),
            "joint_action": int(joint_action),
            "reward": float(reward),
            "next_state": self.clone_state_to_cpu(next_state),
            "done": float(done)
        })

    def clear(self):
        self.records.clear()

    def __len__(self):
        return len(self.records)

    def as_batch(self):
        if not self.records:
            raise ValueError("RolloutBuffer 为空。")

        return {
            "states": [item["state"] for item in self.records],
            "joint_actions": [item["joint_action"] for item in self.records],
            "rewards": [item["reward"] for item in self.records],
            "next_states": [item["next_state"] for item in self.records],
            "dones": [item["done"] for item in self.records]
        }

# ---------------------- GNN 编码器 & Agent ---------------------- #
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=NODE_FEATURE_DIM, hid=32, out_dim=32, use_attention=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.out_dim = out_dim

        self.use_attention = use_attention
        self.score = (
            torch.nn.Linear(out_dim, 1)
            if use_attention
            else None
        )

        self.fallback = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim)
        )

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if edge_index is None or edge_index.numel() == 0:
            h = self.fallback(x)
        else:
            h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            h = self.conv2(h, edge_index, edge_weight=edge_weight)

        if self.use_attention:
            scores = self.score(h).squeeze(-1)
            alpha = pyg_softmax(scores, batch)
            g_avg = global_add_pool(alpha.unsqueeze(-1) * h, batch)
        else:
            node_sum = global_add_pool(h, batch)
            node_count = torch.bincount(batch, minlength=node_sum.size(0)).to(h.dtype).unsqueeze(-1)
            g_avg = node_sum / node_count.clamp_min(1.0)

        g_max = global_max_pool(h, batch)
        return torch.cat([g_avg, g_max], dim=-1)


def load_pretrained_checkpoint(agent, file_path):
    """Load network weights while keeping freshly initialized online optimizers."""
    checkpoint_path = Path(file_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"预训练模型不存在: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        architecture = checkpoint.get("architecture")
        if architecture is not None and architecture != CHECKPOINT_ARCHITECTURE:
            raise ValueError(
                f"checkpoint 架构为 {architecture!r}，"
                f"当前需要 {CHECKPOINT_ARCHITECTURE!r}。"
            )

        feature_names = checkpoint.get("node_feature_names")
        if feature_names is not None and tuple(feature_names) != NODE_FEATURE_NAMES:
            raise ValueError("checkpoint 的节点特征名称或顺序与当前 Online 程序不一致。")

        state_dict = checkpoint["agent_state_dict"]
        global_step = int(checkpoint.get("global_step", 0))
    else:
        state_dict = checkpoint
        global_step = 0

    try:
        agent.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "预训练模型与当前23维 GNN + 单 Actor(25算子对) + 单 Critic 架构不兼容。"
        ) from exc

    return global_step


def save_online_checkpoint(agent, file_path, global_step=0):
    checkpoint_path = Path(file_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": CHECKPOINT_ARCHITECTURE,
        "node_feature_names": NODE_FEATURE_NAMES,
        "action_mode": "single_actor_25_pairs",
        "agent_state_dict": agent.state_dict(),
        "opt_enc_state_dict": agent.opt_enc.state_dict(),
        "opt_critic_state_dict": agent.opt_critic.state_dict(),
        "opt_joint_state_dict": agent.opt_joint.state_dict(),
        "global_step": int(global_step)
    }, checkpoint_path)


def scale_online_learning_rates(agent, scale):
    if scale <= 0:
        raise ValueError('--pretrained_lr_scale 必须大于 0。')

    for optimizer in (
        agent.opt_enc,
        agent.opt_critic,
        agent.opt_joint
    ):
        for param_group in optimizer.param_groups:
            param_group["lr"] *= scale


class SingleACAgent(torch.nn.Module):
    def __init__(
            self,
            enc_dim=64,
            hid=64,
            gamma_repair=0.9,
            critic_coef=1.0
    ):
        super().__init__()
        self.n_destroy = 5
        self.n_repair = 5
        self.n_joint = self.n_destroy * self.n_repair  # 25维联合空间
        self.extra_state_dim = 12 + GLOBAL_TIME_STATE_DIM
        self.min_action_prob = 0.008

        self.encoder = GraphEncoder(
            in_dim=NODE_FEATURE_DIM,
            hid=hid,
            out_dim=enc_dim,
            use_attention=True
        ).to(device)

        encoded_dim = self.encoder.out_dim * 2
        complete_state_dim = encoded_dim + self.extra_state_dim

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(complete_state_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, 1)
        ).to(device)

        # 1. 替换为单个 25 维联合 Actor
        self.actor_joint = torch.nn.Sequential(
            torch.nn.Linear(complete_state_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, self.n_joint)
        ).to(device)

        # 2. 合并优化器，使用 opt_joint 替换原有的 opt_ad 和 opt_ar
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=3e-5, weight_decay=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.opt_joint = torch.optim.Adam(self.actor_joint.parameters(), lr=1e-4, weight_decay=1e-4)

        self.gamma_destroy = 1.0
        self.gamma_repair = gamma_repair
        self.critic_coef = critic_coef

    # 3. 新增：联合采样控制接口
    def act_joint(self, state, deterministic=False):
        logits = self.actor_joint(state)
        distribution = self._policy_distribution(logits)

        if deterministic:
            action_tensor = torch.argmax(
                distribution.probs,
                dim=-1
            )
        else:
            action_tensor = distribution.sample()

        probabilities = distribution.probs

        return (
            int(action_tensor.item()),
            probabilities.detach().cpu().squeeze(0).numpy()
        )

    @staticmethod
    def _expand_scalar_feature(value, batch_size, dtype, target_device):
        tensor = torch.as_tensor(value, dtype=dtype, device=target_device).view(-1, 1)
        if tensor.size(0) == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, 1)
        if tensor.size(0) != batch_size:
            raise ValueError(f"状态特征数量 {tensor.size(0)} 与图数量 {batch_size} 不一致。")
        return tensor

    def build_state(
            self,
            graph,
            stage,
            destroy_ratio,
            destroy_id,
            progress,
            temperature_ratio,
            best_gap,
            stagnation_ratio
    ):
        graph_embedding = self.encoder(graph)
        batch_size = graph_embedding.size(0)
        dtype = graph_embedding.dtype
        target_device = graph_embedding.device

        stage_tensor = torch.as_tensor(stage, dtype=torch.long, device=target_device).view(-1)
        if stage_tensor.numel() == 1 and batch_size > 1:
            stage_tensor = stage_tensor.expand(batch_size)
        if stage_tensor.numel() != batch_size:
            raise ValueError("stage 数量与图 Batch 数量不一致。")

        stage_one_hot = F.one_hot(stage_tensor.clamp(0, 1), num_classes=2).to(dtype)
        ratio_tensor = self._expand_scalar_feature(destroy_ratio, batch_size, dtype, target_device)

        destroy_id_tensor = torch.as_tensor(destroy_id, dtype=torch.long, device=target_device).view(-1)
        if destroy_id_tensor.numel() == 1 and batch_size > 1:
            destroy_id_tensor = destroy_id_tensor.expand(batch_size)
        if destroy_id_tensor.numel() != batch_size:
            raise ValueError("destroy_id 数量与图 Batch 数量不一致。")

        valid_destroy_mask = (destroy_id_tensor >= 0).to(dtype).unsqueeze(-1)
        safe_destroy_id = destroy_id_tensor.clamp(0, self.n_destroy - 1)
        destroy_one_hot = F.one_hot(safe_destroy_id, num_classes=self.n_destroy).to(dtype)
        destroy_one_hot = destroy_one_hot * valid_destroy_mask

        progress_tensor = self._expand_scalar_feature(progress, batch_size, dtype, target_device)
        temperature_tensor = self._expand_scalar_feature(temperature_ratio, batch_size, dtype, target_device)
        best_gap_tensor = self._expand_scalar_feature(best_gap, batch_size, dtype, target_device)
        stagnation_tensor = self._expand_scalar_feature(stagnation_ratio, batch_size, dtype, target_device)

        graph_batch = getattr(graph, 'batch', None)
        if graph_batch is None:
            graph_batch = torch.zeros(
                graph.x.size(0),
                dtype=torch.long,
                device=target_device
            )
        non_depot_mask = (1.0 - graph.x[:, FEATURE_DEPOT_FLAG]).unsqueeze(-1)
        served_mask = graph.x[:, FEATURE_SERVED_FLAG].unsqueeze(-1)
        retained_mask = non_depot_mask * served_mask
        removed_mask = non_depot_mask * (1.0 - served_mask)

        retained_time_features = graph.x[:, [
            FEATURE_CURRENT_WAIT_LOCAL,
            FEATURE_LOCAL_TIME_PRESSURE,
            FEATURE_DOWNSTREAM_PRESSURE
        ]]
        retained_sum = global_add_pool(
            retained_time_features * retained_mask,
            graph_batch
        )
        retained_count = global_add_pool(
            retained_mask,
            graph_batch
        ).clamp_min(1.0)

        removed_time_features = graph.x[:, [
            FEATURE_READY_GLOBAL,
            FEATURE_DIRECT_SLACK_LOCAL
        ]]
        removed_sum = global_add_pool(
            removed_time_features * removed_mask,
            graph_batch
        )
        removed_count = global_add_pool(
            removed_mask,
            graph_batch
        ).clamp_min(1.0)

        tw_global_state = torch.cat(
            [
                retained_sum / retained_count,
                removed_sum / removed_count
            ],
            dim=-1
        )

        extra_state = torch.cat(
            [
                stage_one_hot,
                ratio_tensor,
                destroy_one_hot,
                progress_tensor,
                temperature_tensor,
                best_gap_tensor,
                stagnation_tensor,
                tw_global_state
            ],
            dim=-1
        )
        return torch.cat([graph_embedding, extra_state], dim=-1)

    def encode_state_records(self, state_records):
        graphs, stages, ratios, destroy_ids = [], [], [], []
        progresses, temperature_ratios, best_gaps, stagnation_ratios = [], [], [], []

        for record in state_records:
            g, s, dr, di, p, tr, bg, sr = record
            graphs.append(g)
            stages.append(int(s))
            ratios.append(float(dr))
            destroy_ids.append(int(di))
            progresses.append(float(p))
            temperature_ratios.append(float(tr))
            best_gaps.append(float(bg))
            stagnation_ratios.append(float(sr))

        graph_batch = Batch.from_data_list(graphs).to(device)
        return self.build_state(
            graph=graph_batch,
            stage=stages,
            destroy_ratio=ratios,
            destroy_id=destroy_ids,
            progress=progresses,
            temperature_ratio=temperature_ratios,
            best_gap=best_gaps,
            stagnation_ratio=stagnation_ratios
        )

        # 增加：统一的带保底探索分布生成方法
    def _policy_distribution(self, logits):
        raw_probs = F.softmax(logits, dim=-1)

        num_actions = raw_probs.size(-1)
        p_min = min(
            self.min_action_prob,
            (1.0 - 1e-6) / num_actions
        )

        probs = (
                (1.0 - num_actions * p_min) * raw_probs
                + p_min
        )

        return torch.distributions.Categorical(probs=probs)

    # 替换：去除 @staticmethod 的动作选择方法
    def _select_action(self, logits, deterministic=False):
        distribution = self._policy_distribution(logits)

        if deterministic:
            action_tensor = torch.argmax(
                distribution.probs,
                dim=-1
            )
        else:
            action_tensor = distribution.sample()

        probabilities = distribution.probs

        return (
            int(action_tensor.item()),
            probabilities.detach().cpu().squeeze(0).numpy()
        )



    @staticmethod
    def _normalize_advantage(advantage):
        if advantage.numel() <= 1:
            return advantage

        centered = advantage - advantage.mean()

        # 不再除以当前 batch 的微小标准差
        return centered.clamp(-1.0, 1.0)

    @staticmethod
    def _grad_norm(parameters):
        squared_norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                norm = parameter.grad.detach().norm(2)
                squared_norm += norm.item() ** 2
        return squared_norm ** 0.5

    def update_rollout(self, rollout_buffer, entropy_coef):
        if len(rollout_buffer) == 0:
            return None

        batch = rollout_buffer.as_batch()

        # 1. 批构图特征编码
        states = self.encode_state_records(batch["states"])
        joint_actions = torch.tensor(batch["joint_actions"], dtype=torch.long, device=device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float, device=device).view(-1, 1)
        dones = torch.tensor(batch["dones"], dtype=torch.float, device=device).view(-1, 1)

        # 2. 单阶段价值评估
        values = self.critic(states)

        with torch.no_grad():
            next_states = self.encode_state_records(batch["next_states"])
            target_value_next = self.critic(next_states)

            common_target = (
                    rewards
                    + self.gamma_repair
                    * (1.0 - dones)
                    * target_value_next
            )

        # 3. 单阶段目标 Huber 损失计算
        td_error = common_target - values
        critic_loss = F.smooth_l1_loss(values, common_target)
        advantage = self._normalize_advantage(td_error.detach()).clamp(-5.0, 5.0)

        # 4. 联合策略梯度计算与熵正则化
        joint_logits = self.actor_joint(states)
        joint_distribution = self._policy_distribution(joint_logits)
        joint_log_prob = joint_distribution.log_prob(joint_actions).view(-1, 1)

        raw_entropy = joint_distribution.entropy().mean()
        # 计算归一化熵进行公平评价
        normalized_entropy = raw_entropy / math.log(self.n_joint)

        actor_loss = -(joint_log_prob * advantage).mean() - entropy_coef * raw_entropy
        total_loss = self.critic_coef * critic_loss + actor_loss

        # 5. 反向传播更新
        self.opt_enc.zero_grad()
        self.opt_critic.zero_grad()
        self.opt_joint.zero_grad()

        total_loss.backward()

        encoder_grad_norm = self._grad_norm(self.encoder.parameters())
        critic_grad_norm = self._grad_norm(self.critic.parameters())
        joint_grad_norm = self._grad_norm(self.actor_joint.parameters())

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_joint.parameters(), max_norm=1.0)

        self.opt_enc.step()
        self.opt_critic.step()
        self.opt_joint.step()

        statistics = {
            "total_loss": float(total_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "raw_entropy": float(raw_entropy.detach().item()),
            "normalized_entropy": float(normalized_entropy.detach().item()),
            "td_abs": float(td_error.detach().abs().mean().item()),
            "encoder_grad_norm": encoder_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "destroy_grad_norm": joint_grad_norm,  # 映射原统计字段
            "repair_grad_norm": 0.0
        }

        rollout_buffer.clear()
        return statistics

# ---------------------- 状态 & 熵系数辅助 ---------------------- #
def make_state_record(
        graph,
        stage,
        destroy_ratio,
        destroy_id,
        progress,
        temperature_ratio,
        best_gap,
        stagnation_ratio
):
    return (
        graph,
        int(stage),
        float(destroy_ratio),
        int(destroy_id),
        float(np.clip(progress, 0.0, 1.0)),
        float(np.clip(temperature_ratio, 0.0, 1.0)),
        float(np.clip(best_gap, 0.0, 5.0)),
        float(np.clip(stagnation_ratio, 0.0, 1.0))
    )

def encode_single_state(agent, state_record):
    (
        graph,
        stage,
        destroy_ratio,
        destroy_id,
        progress,
        temperature_ratio,
        best_gap,
        stagnation_ratio
    ) = state_record

    return agent.build_state(
        graph=graph,
        stage=stage,
        destroy_ratio=destroy_ratio,
        destroy_id=destroy_id,
        progress=progress,
        temperature_ratio=temperature_ratio,
        best_gap=best_gap,
        stagnation_ratio=stagnation_ratio
    )

def get_entropy_coef(global_progress, entropy_start=0.01, entropy_end=0.001):
    global_progress = float(np.clip(global_progress, 0.0, 1.0))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * global_progress))
    return entropy_end + (entropy_start - entropy_end) * cosine_factor

# ---------------------- 奖励函数重构 ---------------------- #
def calculate_transition_reward(
        current_obj,
        candidate_obj,
        best_obj_before,
        best_obj_after,
        initial_obj,
        accepted,
        current_weight=5.0,
        best_weight=50.0,
        reject_penalty=0.01
):
    scale = max(abs(float(initial_obj)), 1e-6)

    # 评价候选解本身的质量，而不是回滚后的质量
    candidate_gain = (
        float(current_obj) - float(candidate_obj)
    ) / scale
    if candidate_gain < 0:
        # 当解变差（负值）时，可以打折
        penalty_gain = candidate_gain * 1
    else:
        # 当解变好（正值）时，保持原样，不打折
        penalty_gain = candidate_gain
    best_gain = (
        float(best_obj_before) - float(best_obj_after)
    ) / scale

    reward = (
        current_weight * penalty_gain
        + best_weight * max(0.0, best_gain)
    )

    # 如果差解被拒绝，追加微小的额外拒绝惩罚，鼓励生成易被接受的解
    if not accepted and candidate_obj > current_obj:
        reward -= reject_penalty

    return float(np.clip(reward, -1.0, 1.0))

# ---------------------- 绘图辅助 ---------------------- #
def plotObj(obj_list):
    try:
        plt.figure()
        plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
        plt.xlabel('Iterations')
        plt.ylabel('Obj Value')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

def plotRoutes(model: Model):
    try:
        plt.figure()
        for route in model.best_sol.route_list:
            xs, ys = [], []
            for node_no in route:
                xs.append(model.demand_dict[node_no].x_coord)
                ys.append(model.demand_dict[node_no].y_coord)
            plt.plot(xs, ys, marker='s', linewidth=0.8)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass




# ---------------------- 主函数入口 ---------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-AC 25-Pair Online ALNS-VRPTW Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='单个实例(CSV)文件路径')
    parser.add_argument('--result_path', type=str, required=True, help='结果文件路径')
    parser.add_argument('--knn_k', type=int, default=5, help='kNN邻居数')
    parser.add_argument('--full_graph_knn', action='store_true', help='使用全连接图替代kNN')
    parser.add_argument('--weight_mode', choices=['inv', 'dist'], default='inv', help='边权计算模式')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--rollout_size', type=int, default=8, help='每次在线更新使用的 rollout 步数')
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help='兼容的Single-AC预训练checkpoint；仅加载网络权重，随后继续在线更新'
    )
    parser.add_argument(
        '--pretrained_lr_scale',
        type=float,
        default=1,
        help='加载预训练模型后，Online版各学习率的缩放倍数（默认1）'
    )
    parser.add_argument(
        '--online_model_path',
        type=str,
        default=None,
        help='可选：保存本次在线适配后的checkpoint；不会覆盖预训练模型'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=None,
        help='ALNS迭代次数；默认 max(50 * 客户数, 2000)'
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if (
        args.pretrained_model_path is not None
        and args.online_model_path is not None
        and Path(args.pretrained_model_path).resolve()
        == Path(args.online_model_path).resolve()
    ):
        raise ValueError(
            '--online_model_path 不能与 --pretrained_model_path 相同，'
            '以免覆盖原始预训练模型。'
        )

    gamma = 0.99

    # 初始化单实例相关的环境
    model = Model()
    readCsvFile(args.instance_path, model)
    calDistanceMatrix(model)

    nodes = [model.depot] + [model.demand_dict[nid] for nid in model.node_id_list]

    sol = Sol()
    sol.node_no_seq = genInitialSol(model)

    (
        sol.obj,
        sol.route_list,
        sol.route_distance,
        sol.timetable_list
    ) = calObj(sol.node_no_seq, model)
    if not math.isfinite(sol.obj):
        raise ValueError("无法构造满足容量和时间窗约束的初始解。")

    model.best_sol = copy.deepcopy(sol)

    base_edge_index, base_edge_weight = precompute_base_graph(
        nodes, model, k_nn=args.knn_k, use_full_graph_knn=args.full_graph_knn, weight_mode=args.weight_mode
    )

    agent = SingleACAgent(
        enc_dim=64,
        hid=64,
        gamma_repair=gamma,
        critic_coef=1.0
    ).to(device)

    pretrained_global_step = 0
    if args.pretrained_model_path is not None:
        pretrained_global_step = load_pretrained_checkpoint(
            agent,
            args.pretrained_model_path
        )
        scale_online_learning_rates(agent, args.pretrained_lr_scale)
        print(
            f"已加载预训练模型: {args.pretrained_model_path} | "
            f"global_step={pretrained_global_step} | "
            f"在线学习率缩放={args.pretrained_lr_scale:g}"
        )

    agent.train()

    default_num_steps = max(50 * model.number_of_demands, 2000)
    num_steps = args.num_steps if args.num_steps is not None else default_num_steps
    if num_steps <= 0:
        raise ValueError('--num_steps 必须为正整数。')
    if args.rollout_size <= 0:
        raise ValueError('--rollout_size 必须为正整数。')

    initialization_mode = (
        "预训练模型热启动"
        if args.pretrained_model_path is not None
        else "随机初始化"
    )
    print(f"Single-AC-VRPTW 在线更新求解：{initialization_mode}，全量迭代 {num_steps} 步。")

    rollout_buffer = RolloutBuffer()
    rollout_size = args.rollout_size
    model.T0 = 50
    initial_temperature = float(model.T0)
    stagnation_steps = 0
    initial_obj = float(sol.obj)

    recent_joint_probs = []
    window_size = 200

    history_best_obj = [sol.obj]
    return_list = []
    start_time = time.time()
    all_iterations = []

    model.phi = (0.05 / initial_temperature) ** (1.0 / max(1, num_steps))

    t_graph_build = 0.0
    t_inference = 0.0
    t_training = 0.0
    online_update_count = 0

    # ---------------------- 单算例主循环 (联合动作消融版) ---------------------- #
    for it in range(num_steps):
        iteration_progress = it / max(1, num_steps - 1)
        temperature_ratio = model.T0 / max(initial_temperature, 1e-6)
        current_best_gap = (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6)
        stagnation_ratio = stagnation_steps / max(1, num_steps)

        # =====================================================
        # 1. 当前完整状态 ( stage=0, destroy_ratio=0.0, destroy_id=-1 对齐控制变量 )
        # =====================================================
        t0 = time.perf_counter()

        g_s = build_graph(
            sol,
            nodes,
            model,
            base_edge_index,
            base_edge_weight,
            removed_set=None,
            include_path_edges=True
        )

        full_state_record = make_state_record(
            graph=g_s,
            stage=0,
            destroy_ratio=0.0,
            destroy_id=-1,
            progress=iteration_progress,
            temperature_ratio=temperature_ratio,
            best_gap=current_best_gap,
            stagnation_ratio=stagnation_ratio
        )

        with torch.no_grad():
            full_state_tensor = encode_single_state(agent, full_state_record)

        t_graph_build += (time.perf_counter() - t0)

        # =====================================================
        # 2. Actor 决策 (25维联合动作空间)
        # =====================================================
        t0 = time.perf_counter()
        deterministic = False

        a_joint, joint_prob = agent.act_joint(
            full_state_tensor,
            deterministic=deterministic
        )

        t_inference += (time.perf_counter() - t0)

        # 解码至环境
        a_d = a_joint // 5
        a_r = a_joint % 5

        model.d_select[a_d] += 1
        model.r_select[a_r] += 1

        # =====================================================
        # 3. 环境执行破坏与恢复 (彻底跳过中间部分状态的特征图建构与推理)
        # =====================================================
        destroyed_routes, remove_list = doDestroy_by_id(a_d, model, sol, it, num_steps)
        new_sol = doRepair_by_id(a_r, destroyed_routes, remove_list, model)

        # =====================================================
        # 4. 评估与接受判断
        # =====================================================
        current_obj_before = float(sol.obj)
        best_obj_before = float(model.best_sol.obj)
        candidate_obj = float(new_sol.obj)

        if candidate_obj < current_obj_before:
            accepted = True
        else:
            acceptance_probability = math.exp(
                (current_obj_before - candidate_obj)
                / max(model.T0, 1e-6)
            )
            accepted = acceptance_probability >= random.random()

        if accepted:
            sol = copy.deepcopy(new_sol)

        if candidate_obj < model.best_sol.obj:
            model.best_sol = copy.deepcopy(new_sol)
            found_new_best = True
            stagnation_steps = 0
        else:
            found_new_best = False
            stagnation_steps += 1

        next_obj = float(sol.obj)
        best_obj_after = float(model.best_sol.obj)

        repair_reward = calculate_transition_reward(
            current_obj=current_obj_before,
            candidate_obj=candidate_obj,
            best_obj_before=best_obj_before,
            best_obj_after=best_obj_after,
            initial_obj=initial_obj,
            accepted=accepted
        )

        # =====================================================
        # 5. 实际下一完整状态构建
        # =====================================================
        if accepted:
            next_graph = build_graph(
                sol,
                nodes,
                model,
                base_edge_index,
                base_edge_weight,
                removed_set=None,
                include_path_edges=True
            )
        else:
            next_graph = g_s

        model.T0 *= model.phi

        next_progress = (it + 1) / max(1, num_steps - 1)
        next_temperature_ratio = model.T0 / max(initial_temperature, 1e-6)
        next_best_gap = (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6)
        next_stagnation_ratio = stagnation_steps / max(1, num_steps)
        is_terminal = float(it == num_steps - 1)

        next_full_state_record = make_state_record(
            graph=next_graph,
            stage=0,
            destroy_ratio=0.0,
            destroy_id=-1,
            progress=next_progress,
            temperature_ratio=next_temperature_ratio,
            best_gap=next_best_gap,
            stagnation_ratio=next_stagnation_ratio
        )

        # =====================================================
        # 6. 保存 on-policy rollout (单阶段结构存储)
        # =====================================================
        rollout_buffer.add(
            state=full_state_record,
            joint_action=a_joint,
            reward=repair_reward,
            next_state=next_full_state_record,
            done=is_terminal
        )

        return_list.append(repair_reward)
        recent_joint_probs.append(joint_prob)

        if len(recent_joint_probs) > window_size:
            recent_joint_probs = recent_joint_probs[-window_size:]

        # =====================================================
        # 7. Rollout 触发参数更新 (单阶段梯度反向更新)
        # =====================================================
        should_update = len(rollout_buffer) >= rollout_size or is_terminal > 0.5

        if should_update:
            t0 = time.perf_counter()
            entropy_coef = get_entropy_coef(
                global_progress=iteration_progress,
                entropy_start=0.01,
                entropy_end=0.001
            )

            update_statistics = agent.update_rollout(
                rollout_buffer,
                entropy_coef=entropy_coef
            )
            t_training += (time.perf_counter() - t0)

            if update_statistics is not None:
                online_update_count += 1
                # Keep per-update diagnostics disabled so timing matches DAC.

        history_best_obj.append(model.best_sol.obj)

        # ---------------------- 打印与记录 ---------------------- #
        if (it + 1) % 200 == 0:
            avg_joint_prob = np.mean(recent_joint_probs, axis=0)
            joint_matrix = avg_joint_prob.reshape(5, 5)

            # Keep rolling diagnostics disabled so timing matches DAC.

        if (it + 1) % 10 == 0:
            all_iterations.append({
                "iteration": it + 1,
                "file_path": args.instance_path,
                "best_obj": round(model.best_sol.obj, 2),
                "current_obj": round(sol.obj, 2),
                "repair_reward": round(repair_reward, 6),
                "accepted": bool(accepted),
                "new_best": bool(found_new_best),
                "joint_pair_probabilities": [
                    round(float(probability), 6)
                    for probability in joint_prob
                ],
                "run_time": round(time.time() - start_time, 2),
                "destroy_op1": float(model.d_select[0]),
                "destroy_op2": float(model.d_select[1]),
                "destroy_op3": float(model.d_select[2]),
                "destroy_op4": float(model.d_select[3]),
                "destroy_op5": float(model.d_select[4]),
                "repair_op1": float(model.r_select[0]),
                "repair_op2": float(model.r_select[1]),
                "repair_op3": float(model.r_select[2]),
                "repair_op4": float(model.r_select[3]),
                "repair_op5": float(model.r_select[4])
            })



    # ---------------------- 统计时间开销 ---------------------- #
    run_time = time.time() - start_time
    nn_total_extra = t_graph_build + t_inference + t_training
    nn_percentage = (nn_total_extra / max(run_time, 1e-6)) * 100

    excel_record = {
        "Instance": Path(args.instance_path).name,
        "Best_Obj": round(model.best_sol.obj, 2),
        "Total_Time(s)": round(run_time, 2),
        "NN_Graph_Build(s)": round(t_graph_build, 3),
        "NN_Inference(s)": round(t_inference, 3),
        "NN_Training(s)": round(t_training, 3),
        "NN_Total_Extra(s)": round(nn_total_extra, 3),
        "NN_Overhead_Ratio(%)": round(nn_percentage, 2),
        "Problem": "VRPTW",
        "Node_Feature_Dim": NODE_FEATURE_DIM,
        "Action_Mode": "single_actor_25_pairs",
        "Initialization": initialization_mode,
        "Pretrained_Model": args.pretrained_model_path or "",
        "Pretrained_LR_Scale": (
            args.pretrained_lr_scale
            if args.pretrained_model_path is not None
            else ""
        ),
        "Online_Updates": online_update_count,
        "Rollout_Size": rollout_size,
        "Iterations": num_steps,
        "Device": str(device)
    }

    # ---------------------- 收尾与文件输出 ---------------------- #
    parent_path = Path(args.result_path).parent
    parent_path.mkdir(parents=True, exist_ok=True)
    parts = Path(args.result_path).parts
    file_name = Path(args.result_path).stem

    csv_summary_path = parent_path / "Single_AC_ALNS_VRPTW_Summary_Results.csv"
    save_to_csv(csv_summary_path, excel_record)

    log_dir = parent_path / 'LOG'
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f'{file_name}_iteration_log_{round(model.best_sol.obj, 2)}_{round(run_time, 2)}.json'
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_iterations, f, indent=4, ensure_ascii=False)

    try:
        result_path = parent_path / f'{parts[-2]}_result.json'
        safe_append_to_json(result_path, file_name, round(model.best_sol.obj, 2), round(run_time, 2))
    except IndexError:
        print("Warn: result aggregation path not found; skip.")

    if args.online_model_path is not None:
        save_online_checkpoint(
            agent,
            args.online_model_path,
            global_step=pretrained_global_step + num_steps
        )
        print(f"在线适配模型已保存至: {args.online_model_path}")

    print("\n--- Final Statistics ---")
    print(f"Best Objective Found: {model.best_sol.obj:.2f}")
    print(f"Total Run Time: {run_time:.2f}s")
    print(f"Online Parameter Updates: {online_update_count}")
    print(f"Neural Network Overhead Ratio: {nn_percentage:.2f}%")
    print(f"Destroy Op 1 Selections (Random):  {model.d_select[0]}")
    print(f"Destroy Op 2 Selections (Worst):   {model.d_select[1]}")
    print(f"Destroy Op 3 Selections (Shaw):    {model.d_select[2]}")
    print(f"Destroy Op 4 Selections (SISR):    {model.d_select[3]}")
    print(f"Destroy Op 5 Selections (CritTW):  {model.d_select[4]}")
    print(f"Repair  Op 1 Selections (RndBest): {model.r_select[0]}")
    print(f"Repair  Op 2 Selections (Regret):  {model.r_select[1]}")
    print(f"Repair  Op 3 Selections (Cheap):   {model.r_select[2]}")
    print(f"Repair  Op 4 Selections (Regret3): {model.r_select[3]}")
    print(f"Repair  Op 5 Selections (TWDelay): {model.r_select[4]}")
    print("------------------------\n")
