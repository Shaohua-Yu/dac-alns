# -*- coding: utf-8 -*-
# DAC-ALNS (统一单头 Critic & 12维动态节点特征 & 双阶段 On-Policy DUEL-AC 版)

import os
import csv
import math
import time
import json
import copy
import random
import argparse
from pathlib import Path

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

class Node:
    def __init__(self):
        self.id = 0
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.demand = 0.0

class Model:
    def __init__(self):
        self.best_sol = None
        self.node_id_list = []  # 不含depot
        self.demand_dict = {}  # id->Node（含depot）
        self.depot = None
        self.number_of_demands = 0
        self.vehicle_cap = 100
        self.distance_matrix = {}
        self.distance_matrix_dict = {}
        self.demands_arr = None
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
    with open(demand_file, 'r') as f:
        reader = csv.DictReader(f)
        v_cap_assigned = False
        for row in reader:
            if not v_cap_assigned:
                model.vehicle_cap = float(row['v_cap'])
                v_cap_assigned = True
            node = Node()
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            if node.demand == 0:
                node.id = 0
                model.depot = node
            else:
                node.id = int(float(row['id'])) - 1
                model.node_id_list.append(node.id)
            model.demand_dict[node.id] = node
    model.number_of_demands = len(model.demand_dict) - 1

def calDistanceMatrix(model):
    ids = [model.depot.id] + model.node_id_list
    coords = np.array([[model.demand_dict[i].x_coord,
                        model.demand_dict[i].y_coord] for i in ids],
                      dtype=np.float32)

    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(-1, dtype=np.float32))

    model.distance_matrix = dist_mat
    model.id2idx = {id_: idx for idx, id_ in enumerate(ids)}
    model.idx2id = ids

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            model.distance_matrix_dict[(id_i, id_j)] = dist_mat[i, j]

    # 初始化需求快速查询
    max_id = max(model.demand_dict.keys())
    model.demands_arr = np.zeros(max_id + 1, dtype=np.float32)
    for nid, node in model.demand_dict.items():
        model.demands_arr[nid] = node.demand

    # 预计算 Shaw 相似度矩阵
    n = len(ids)
    sim_mat = np.zeros((n, n), dtype=np.float32)
    max_d = np.max(dist_mat) + 1e-9
    max_q = np.max(model.demands_arr) + 1e-9
    for u in range(n):
        for v in range(n):
            dist_score = dist_mat[u, v] / max_d
            demand_score = abs(model.demands_arr[u] - model.demands_arr[v]) / max_q
            sim_mat[u, v] = 0.8 * dist_score + 0.2 * demand_score
    model.similarity_matrix = sim_mat

def genInitialSol(model: Model):
    unvisited = copy.deepcopy(model.node_id_list)
    if not unvisited:
        return []
    seq = []
    curr = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: model.distance_matrix[model.id2idx[curr], model.id2idx[x]])
        seq.append(nearest)
        unvisited.remove(nearest)
        curr = nearest
    return seq

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

def _split_dp_core_py(seq, dem_arr, Q, dist_mat, depot_idx):
    n = seq.shape[0]
    d0 = np.empty(n, dtype=np.float32)
    dr = np.empty(n, dtype=np.float32)
    for t in range(n):
        idx = seq[t]
        d0[t] = dist_mat[depot_idx, idx]
        dr[t] = dist_mat[idx, depot_idx]

    F = np.full(n + 1, np.inf, dtype=np.float32)
    pred = np.zeros(n + 1, dtype=np.int32)
    F[0] = 0.0

    for i in range(1, n + 1):
        load = 0.0
        cost = 0.0
        for j in range(i, 0, -1):
            load += dem_arr[j - 1]
            if load > Q:
                break
            if j == i:
                cost = d0[j - 1] + dr[j - 1]
            else:
                prev_idx = seq[j]
                cur_idx = seq[j - 1]
                cost = cost \
                       - dist_mat[prev_idx, depot_idx] \
                       + dist_mat[prev_idx, cur_idx] \
                       + dist_mat[cur_idx, depot_idx]

            new_cost = F[j - 1] + cost
            if new_cost < F[i]:
                F[i] = new_cost
                pred[i] = j - 1
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
        dem_arr,
        model.vehicle_cap,
        model.distance_matrix,
        depot_idx=0
    )

    routes_idx = []
    i = len(seq_idx)
    while i > 0:
        j = int(pred[i]) + 1
        routes_idx.append(seq_idx[j - 1:i])
        i = int(pred[i])

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

def calObj(node_no_seq, model):
    route_list, total_distance = splitRoutes(node_no_seq, model)
    if not route_list:
        return float('inf'), [], []
    route_distance = []
    for route in route_list:
        idx = [model.id2idx[id_] for id_ in route]
        d = model.distance_matrix[idx[:-1], idx[1:]].sum()
        route_distance.append(float(d))

    return float(total_distance), route_list, route_distance


def apply_destroy_to_partial_sol_explicit(destroyed_routes, model: Model):
    """
    方案A：直接基于显式残破路线构建 Partial Sol，不调用 DP Split
    """
    part = Sol()
    part.route_list = destroyed_routes

    # 累加各条残破路线的实际距离
    route_distances = []
    for route in destroyed_routes:
        idx = [model.id2idx[id_] for id_ in route]
        d = model.distance_matrix[idx[:-1], idx[1:]].sum()
        route_distances.append(float(d))

    part.route_distance = route_distances
    part.obj = sum(route_distances)

    # 一维序列仅用作状态记录（如 GNN 输入），通过扁平化路线直接获得
    part.node_no_seq = [c for r in destroyed_routes for c in r[1:-1]]
    return part

# ==================== 5x5 算子库 ==================== #

# --- 1. R1: 随机客户移除 ---
def destroy_strategy_op1(routes, k, dist_mat, demands, capacity):
    all_customers = [c for r in routes for c in r[1:-1]]
    if not all_customers or k <= 0:
        return [r[:] for r in routes], []
    k = min(k, len(all_customers))
    removed = random.sample(all_customers, k)
    removed_set = set(removed)
    new_routes = [[c for c in r if c not in removed_set] for r in routes]
    return [r for r in new_routes if len(r) > 2], sorted(removed)

# --- 2. R16: 最差成本客户移除 ---
def destroy_strategy_op2(routes, k, dist_mat, demands, capacity, p_worst=5):
    all_custs = [c for r in routes for c in r[1:-1]]
    if not all_custs or k <= 0:
        return [r[:] for r in routes], []
    k = min(k, len(all_custs))

    cust_costs = []
    for r_idx, r in enumerate(routes):
        for i in range(1, len(r) - 1):
            cost = dist_mat[r[i - 1]][r[i]] + dist_mat[r[i]][r[i + 1]] - dist_mat[r[i - 1]][r[i + 1]]
            cust_costs.append((cost, r[i]))

    cust_costs.sort(key=lambda x: x[0], reverse=True)

    removed = []
    for _ in range(k):
        if not cust_costs:
            break
        idx = int(math.floor(len(cust_costs) * (random.random() ** p_worst)))
        idx = max(0, min(idx, len(cust_costs) - 1))
        _, cust = cust_costs.pop(idx)
        removed.append(cust)

    removed_set = set(removed)
    new_routes = [[c for c in r if c not in removed_set] for r in routes]
    return [r for r in new_routes if len(r) > 2], sorted(removed)

# --- 3. R13: 后验 Shaw 关系移除 ---
def destroy_strategy_op3(routes, k, dist_mat, demands, capacity, similarity_matrix, p_random=6):
    all_custs = [c for r in routes for c in r[1:-1]]
    if not all_custs or k <= 0:
        return [r[:] for r in routes], []
    k = min(k, len(all_custs))

    removed = [random.choice(all_custs)]

    while len(removed) < k:
        ref = random.choice(removed)
        candidates = [c for c in all_custs if c not in removed]
        if not candidates:
            break
        scores = []
        for c in candidates:
            score = similarity_matrix[ref][c]
            scores.append((score, c))
        scores.sort(key=lambda x: x[0])
        idx = int(math.floor(len(scores) * (random.random() ** p_random)))
        idx = max(0, min(idx, len(scores) - 1))
        removed.append(scores[idx][1])

    rem_set = set(removed)
    new_routes = [[c for c in r if c not in rem_set] for r in routes]
    return [r for r in new_routes if len(r) > 2], sorted(removed)

# --- 4. R9/R10: 邻近序列/字符串移除 ---
def destroy_strategy_op4(routes, k, dist_mat, demands, capacity):
    active_routes = [r for r in routes if len(r) > 2]
    if not active_routes or k <= 0:
        return [r[:] for r in routes], []

    total_customers = sum(len(r) - 2 for r in active_routes)
    k = min(k, total_customers)

    new_routes = [r[:] for r in routes]
    removed = []

    shuffled_route_indices = list(range(len(new_routes)))
    random.shuffle(shuffled_route_indices)

    for r_idx in shuffled_route_indices:
        if len(removed) >= k:
            break
        r = new_routes[r_idx]
        if len(r) <= 2:
            continue

        customers = r[1:-1]
        needed = k - len(removed)
        remove_len = min(needed, len(customers))

        start_idx = random.randint(0, len(customers) - remove_len)
        to_remove = customers[start_idx: start_idx + remove_len]
        removed.extend(to_remove)

        rem_set = set(to_remove)
        new_routes[r_idx] = [0] + [c for c in customers if c not in rem_set] + [0]

    return [r for r in new_routes if len(r) > 2], sorted(removed)

# --- 5. R19: 随机多路径整条/块移除 ---
def destroy_strategy_op5(routes, k, dist_mat, demands, capacity):
    active_route_indices = [i for i, r in enumerate(routes) if len(r) > 2]
    if not active_route_indices or k <= 0:
        return [r[:] for r in routes], []

    random.shuffle(active_route_indices)

    removed = []
    new_routes = [r[:] for r in routes]

    for r_idx in active_route_indices:
        if len(removed) >= k:
            break

        r = new_routes[r_idx]
        nodes_in_route = r[1:-1]
        needed = k - len(removed)

        if len(nodes_in_route) <= needed:
            removed.extend(nodes_in_route)
            new_routes[r_idx] = [0, 0]
        else:
            to_remove = nodes_in_route[:needed]
            removed.extend(to_remove)

            rem_set = set(to_remove)
            new_routes[r_idx] = [0] + [c for c in nodes_in_route if c not in rem_set] + [0]

    return [r for r in new_routes if len(r) > 2], sorted(removed)

# --- 6. I5: 随机顺序最佳位置插入 ---
def repair_strategy_op1(routes, removed, dist_mat, demands, capacity):
    if not removed:
        return routes
    new_routes = [r[:] for r in routes if len(r) > 2]
    if not new_routes:
        new_routes = [[0, 0]]

    remaining = list(removed)
    random.shuffle(remaining)
    route_loads = [sum(demands[node] for node in r) for r in new_routes]

    for cust in remaining:
        best_cost = float('inf')
        best_pos = None
        d_c = demands[cust]
        for r_idx, route in enumerate(new_routes):
            if route_loads[r_idx] + d_c <= capacity:
                for i in range(1, len(route)):
                    cost = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][route[i]]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (r_idx, i)
        if best_pos:
            new_routes[best_pos[0]].insert(best_pos[1], cust)
            route_loads[best_pos[0]] += d_c
        else:
            new_routes.append([0, cust, 0])
            route_loads.append(d_c)
    return [r for r in new_routes if len(r) > 2]

# --- 7. I13: 位置后悔值插入 ---
def repair_strategy_op2(routes, removed, dist_mat, demands, capacity):
    if not removed:
        return routes
    new_routes = [r[:] for r in routes if len(r) > 2]
    if not new_routes:
        new_routes = [[0, 0]]

    remaining = list(removed)
    route_loads = [sum(demands[node] for node in r) for r in new_routes]
    max_dist = np.max(dist_mat)

    while remaining:
        regret_candidates = []
        for cust in remaining:
            d_c = demands[cust]
            insertion_costs = []
            for r_idx, route in enumerate(new_routes):
                if route_loads[r_idx] + d_c <= capacity:
                    b_cost = float('inf')
                    b_pos = 1
                    for i in range(1, len(route)):
                        c = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][route[i]]
                        if c < b_cost:
                            b_cost, b_pos = c, i
                    if b_cost != float('inf'):
                        insertion_costs.append((b_cost, r_idx, b_pos))

            insertion_costs.sort(key=lambda x: x[0])
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            elif len(insertion_costs) == 1:
                regret = max_dist - insertion_costs[0][0]
            else:
                regret = -float('inf')

            best_option = insertion_costs[0] if insertion_costs else (dist_mat[0][cust] + dist_mat[cust][0], -1, 1)
            regret_candidates.append({'regret': regret, 'best': best_option, 'cust': cust})

        best_cand = max(regret_candidates, key=lambda x: (x['regret'], -x['best'][0]))
        cost, r_idx, pos = best_cand['best']
        cust = best_cand['cust']

        if r_idx == -1:
            new_routes.append([0, cust, 0])
            route_loads.append(demands[cust])
        else:
            new_routes[r_idx].insert(pos, cust)
            route_loads[r_idx] += demands[cust]
        remaining.remove(cust)

    return [r for r in new_routes if len(r) > 2]

# --- 8. I7: 最小成本优先最佳位置插入 ---
def repair_strategy_op3(routes, removed, dist_mat, demands, capacity):
    if not removed:
        return routes
    new_routes = [r[:] for r in routes if len(r) > 2]
    if not new_routes:
        new_routes = [[0, 0]]

    remaining = list(removed)
    route_loads = [sum(demands[node] for node in r) for r in new_routes]

    while remaining:
        best_cost = float('inf')
        best_cust = None
        target_r_idx = -1
        target_pos = -1
        for cust in remaining:
            d_c = demands[cust]
            for r_idx, route in enumerate(new_routes):
                if route_loads[r_idx] + d_c <= capacity:
                    for i in range(1, len(route)):
                        cost = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][route[i]]
                        if cost < best_cost:
                            best_cost = cost
                            best_cust = cust
                            target_r_idx = r_idx
                            target_pos = i
        if best_cust is None:
            cust = remaining.pop(0)
            new_routes.append([0, cust, 0])
            route_loads.append(demands[cust])
        else:
            new_routes[target_r_idx].insert(target_pos, best_cust)
            route_loads[target_r_idx] += demands[best_cust]
            remaining.remove(best_cust)

    return [r for r in new_routes if len(r) > 2]

# --- 9. I14: 带噪声的位置后悔值插入 ---
def repair_strategy_op4(routes, removed, dist_mat, demands, capacity, noise_level=0.15):
    if not removed:
        return routes
    new_routes = [r[:] for r in routes if len(r) > 2]
    if not new_routes:
        new_routes = [[0, 0]]

    remaining = list(removed)
    route_loads = [sum(demands[node] for node in r) for r in new_routes]
    max_dist = np.max(dist_mat)

    while remaining:
        regret_candidates = []
        for cust in remaining:
            d_c = demands[cust]
            insertion_costs = []
            for r_idx, route in enumerate(new_routes):
                if route_loads[r_idx] + d_c <= capacity:
                    b_cost = float('inf')
                    b_pos = 1
                    for i in range(1, len(route)):
                        c = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][route[i]]
                        if c < b_cost:
                            b_cost, b_pos = c, i
                    if b_cost != float('inf'):
                        noise = (random.random() * 2.0 - 1.0) * noise_level * max_dist
                        perturbed_cost = max(0.0, b_cost + noise)
                        insertion_costs.append((perturbed_cost, r_idx, b_pos))

            insertion_costs.sort(key=lambda x: x[0])
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            elif len(insertion_costs) == 1:
                regret = max_dist - insertion_costs[0][0]
            else:
                regret = -float('inf')

            best_option = insertion_costs[0] if insertion_costs else (dist_mat[0][cust] + dist_mat[cust][0], -1, 1)
            regret_candidates.append({'regret': regret, 'best': best_option, 'cust': cust})

        best_cand = max(regret_candidates, key=lambda x: (x['regret'], -x['best'][0]))
        cost, r_idx, pos = best_cand['best']
        cust = best_cand['cust']

        if r_idx == -1:
            new_routes.append([0, cust, 0])
            route_loads.append(demands[cust])
        else:
            new_routes[r_idx].insert(pos, cust)
            route_loads[r_idx] += demands[cust]
        remaining.remove(cust)

    return [r for r in new_routes if len(r) > 2]

# --- 10. I10: 限制近邻/随机路径插入 ---
def repair_strategy_op5(routes, removed, dist_mat, demands, capacity, k_neighbors=10, limit_routes=3):
    if not removed:
        return routes
    new_routes = [r[:] for r in routes if len(r) > 2]
    if not new_routes:
        new_routes = [[0, 0]]

    remaining = list(removed)
    random.shuffle(remaining)
    route_loads = [sum(demands[node] for node in r) for r in new_routes]

    for cust in remaining:
        d_c = demands[cust]
        num_nodes = dist_mat.shape[0]

        neighbors = sorted(
            [(dist_mat[cust][other], other) for other in range(num_nodes) if other != cust and other != 0],
            key=lambda x: x[0])
        nearest_neighbors = {other for _, other in neighbors[:k_neighbors]}

        candidate_route_indices = []
        for r_idx, route in enumerate(new_routes):
            overlap = sum(1 for node in route if node in nearest_neighbors)
            if overlap > 0:
                candidate_route_indices.append((overlap, r_idx))

        candidate_route_indices.sort(key=lambda x: x[0], reverse=True)
        selected_route_indices = [r_idx for _, r_idx in candidate_route_indices[:limit_routes]]

        if not selected_route_indices:
            all_indices = list(range(len(new_routes)))
            selected_route_indices = random.sample(all_indices, min(limit_routes, len(all_indices)))

        best_cost = float('inf')
        best_r_idx = -1
        best_pos = -1

        for r_idx in selected_route_indices:
            route = new_routes[r_idx]
            if route_loads[r_idx] + d_c <= capacity:
                for i in range(1, len(route)):
                    cost = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][route[i]]
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_pos = i

        if best_r_idx != -1:
            new_routes[best_r_idx].insert(best_pos, cust)
            route_loads[best_r_idx] += d_c
        else:
            new_routes.append([0, cust, 0])
            route_loads.append(d_c)

    return [r for r in new_routes if len(r) > 2]

DESTROY_FUNCS = [destroy_strategy_op1, destroy_strategy_op2, destroy_strategy_op3, destroy_strategy_op4, destroy_strategy_op5]
REPAIR_FUNCS = [repair_strategy_op1, repair_strategy_op2, repair_strategy_op3, repair_strategy_op4, repair_strategy_op5]

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
            capacity=model.vehicle_cap, similarity_matrix=model.similarity_matrix
        )
    else:
        new_routes, removed = DESTROY_FUNCS[did](
            routes, k, dist_mat=model.distance_matrix, demands=model.demands_arr,
            capacity=model.vehicle_cap
        )
    return new_routes,removed


def doRepair_by_id(rid, destroyed_routes, remove_list, model: Model, sol=None, *args, **kwargs):
    """
    方案A：直接在残破路线上执行插入，并直接计算最终距离，杜绝 DP Split
    （兼容版本：支持接收 5 个或更多参数，自动忽略多余的 sol）
    """
    new_routes = REPAIR_FUNCS[rid](
        destroyed_routes, remove_list, dist_mat=model.distance_matrix,
        demands=model.demands_arr, capacity=model.vehicle_cap
    )

    new_sol = Sol()
    new_sol.route_list = [r for r in new_routes if len(r) > 2]

    route_distances = []
    for route in new_sol.route_list:
        idx = [model.id2idx[node_id] for node_id in route]
        d = model.distance_matrix[idx[:-1], idx[1:]].sum()
        route_distances.append(float(d))

    new_sol.route_distance = route_distances
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

# ---------------------- 构图：12维动态特征 ---------------------- #
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

    for idx, node in enumerate(node_list):
        depot_flag = (
            1.0
            if node.id == model.depot.id
            else 0.0
        )

        demand_ratio = (
            float(node.demand) / capacity
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
            float(route_customer_ratio[idx])
        ])

    x = torch.tensor(
        x_data,
        dtype=torch.float,
        device=device
    )

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
    短期 on-policy rollout。
    每条记录包含同一个 ALNS 迭代中的转移状态与奖励。
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
            full_state,
            destroy_action,
            partial_state,
            repair_action,
            repair_reward,
            next_full_state,
            done
    ):
        self.records.append({
            "full_state": self.clone_state_to_cpu(full_state),
            "destroy_action": int(destroy_action),
            "partial_state": self.clone_state_to_cpu(partial_state),
            "repair_action": int(repair_action),
            "repair_reward": float(repair_reward),
            "next_full_state": self.clone_state_to_cpu(next_full_state),
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
            "full_states": [item["full_state"] for item in self.records],
            "destroy_actions": [item["destroy_action"] for item in self.records],
            "partial_states": [item["partial_state"] for item in self.records],
            "repair_actions": [item["repair_action"] for item in self.records],
            "repair_rewards": [item["repair_reward"] for item in self.records],
            "next_full_states": [item["next_full_state"] for item in self.records],
            "dones": [item["done"] for item in self.records]
        }

# ---------------------- GNN 编码器 & Agent ---------------------- #
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=12, hid=32, out_dim=32, use_attention=True):
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

def save_training_checkpoint(agent, file_path, global_step=0):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "agent_state_dict": agent.state_dict(),
        "opt_enc_state_dict": agent.opt_enc.state_dict(),
        "opt_critic_state_dict": agent.opt_critic.state_dict(),
        "opt_ad_state_dict": agent.opt_ad.state_dict(),
        "opt_ar_state_dict": agent.opt_ar.state_dict(),
        "global_step": int(global_step)
    }, file_path)


def load_training_checkpoint(agent, file_path, load_optimizers=True):
    checkpoint = torch.load(file_path, map_location=device)

    # 兼容旧模型
    if not (
        isinstance(checkpoint, dict)
        and "agent_state_dict" in checkpoint
    ):
        agent.load_state_dict(checkpoint)
        return 0

    agent.load_state_dict(checkpoint["agent_state_dict"])

    if load_optimizers:
        if "opt_enc_state_dict" in checkpoint:
            agent.opt_enc.load_state_dict(checkpoint["opt_enc_state_dict"])
        if "opt_critic_state_dict" in checkpoint:
            agent.opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])
        if "opt_ad_state_dict" in checkpoint:
            agent.opt_ad.load_state_dict(checkpoint["opt_ad_state_dict"])
        if "opt_ar_state_dict" in checkpoint:
            agent.opt_ar.load_state_dict(checkpoint["opt_ar_state_dict"])

    return int(checkpoint.get("global_step", 0))

class DACAgent(torch.nn.Module):
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
        self.extra_state_dim = 12
        self.min_action_prob = 0.04

        self.encoder = GraphEncoder(
            in_dim=12,
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

        self.actor_destroy = torch.nn.Sequential(
            torch.nn.Linear(complete_state_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, self.n_destroy)
        ).to(device)

        self.actor_repair = torch.nn.Sequential(
            torch.nn.Linear(complete_state_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, self.n_repair)
        ).to(device)

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=1e-5, weight_decay=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.opt_ad = torch.optim.Adam(self.actor_destroy.parameters(), lr=3e-5, weight_decay=1e-4)
        self.opt_ar = torch.optim.Adam(self.actor_repair.parameters(), lr=3e-5, weight_decay=1e-4)

        self.gamma_destroy = 1.0
        self.gamma_repair = gamma_repair
        self.critic_coef = critic_coef

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

        extra_state = torch.cat(
            [
                stage_one_hot,
                ratio_tensor,
                destroy_one_hot,
                progress_tensor,
                temperature_tensor,
                best_gap_tensor,
                stagnation_tensor
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

    def act_destroy(self, state, deterministic=False):
        logits = self.actor_destroy(state)
        return self._select_action(logits, deterministic=deterministic)

    def act_repair(self, state, deterministic=False):
        logits = self.actor_repair(state)
        return self._select_action(logits, deterministic=deterministic)

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

        full_states = self.encode_state_records(batch["full_states"])
        partial_states = self.encode_state_records(batch["partial_states"])

        destroy_actions = torch.tensor(batch["destroy_actions"], dtype=torch.long, device=device)
        repair_actions = torch.tensor(batch["repair_actions"], dtype=torch.long, device=device)
        repair_rewards = torch.tensor(batch["repair_rewards"], dtype=torch.float, device=device).view(-1, 1)
        dones = torch.tensor(batch["dones"], dtype=torch.float, device=device).view(-1, 1)

        value_full = self.critic(full_states)
        value_partial = self.critic(partial_states)

        with torch.no_grad():
            next_full_states = self.encode_state_records(
                batch["next_full_states"]
            )
            target_value_next_full = self.critic(next_full_states)

            common_target = (
                    repair_rewards
                    + self.gamma_repair
                    * (1.0 - dones)
                    * target_value_next_full
            )

        td_error_destroy = common_target - value_full
        td_error_repair = common_target - value_partial

        critic_loss_destroy = F.smooth_l1_loss(
            value_full,
            common_target
        )

        critic_loss_repair = F.smooth_l1_loss(
            value_partial,
            common_target
        )

        critic_loss = critic_loss_destroy + critic_loss_repair

        advantage_destroy = self._normalize_advantage(td_error_destroy.detach()).clamp(-5.0, 5.0)
        advantage_repair = self._normalize_advantage(td_error_repair.detach()).clamp(-5.0, 5.0)

        destroy_logits = self.actor_destroy(full_states)
        repair_logits = self.actor_repair(partial_states)

        # 替换为：调用上面定义好的 _policy_distribution 生成带有保底概率的分布
        destroy_distribution = self._policy_distribution(
            destroy_logits
        )

        repair_distribution = self._policy_distribution(
            repair_logits
        )

        destroy_log_prob = destroy_distribution.log_prob(destroy_actions).view(-1, 1)
        repair_log_prob = repair_distribution.log_prob(repair_actions).view(-1, 1)

        destroy_entropy = destroy_distribution.entropy().mean()
        repair_entropy = repair_distribution.entropy().mean()

        actor_destroy_loss = -(destroy_log_prob * advantage_destroy).mean() - entropy_coef * destroy_entropy
        actor_repair_loss = -(repair_log_prob * advantage_repair).mean() - entropy_coef * repair_entropy

        total_loss = self.critic_coef * critic_loss + actor_destroy_loss + actor_repair_loss

        self.opt_enc.zero_grad()
        self.opt_critic.zero_grad()
        self.opt_ad.zero_grad()
        self.opt_ar.zero_grad()

        total_loss.backward()

        encoder_grad_norm = self._grad_norm(self.encoder.parameters())
        critic_grad_norm = self._grad_norm(self.critic.parameters())
        destroy_grad_norm = self._grad_norm(self.actor_destroy.parameters())
        repair_grad_norm = self._grad_norm(self.actor_repair.parameters())

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_destroy.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_repair.parameters(), max_norm=1.0)

        self.opt_enc.step()
        self.opt_critic.step()
        self.opt_ad.step()
        self.opt_ar.step()

        statistics = {
            "total_loss": float(total_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "destroy_actor_loss": float(actor_destroy_loss.detach().item()),
            "repair_actor_loss": float(actor_repair_loss.detach().item()),
            "destroy_entropy": float(destroy_entropy.detach().item()),
            "repair_entropy": float(repair_entropy.detach().item()),
            "destroy_td_abs": float(td_error_destroy.detach().abs().mean().item()),
            "repair_td_abs": float(td_error_repair.detach().abs().mean().item()),
            "encoder_grad_norm": encoder_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "destroy_grad_norm": destroy_grad_norm,
            "repair_grad_norm": repair_grad_norm
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

def get_entropy_coef(global_progress, entropy_start=0.05, entropy_end=0.02):
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

# ---------------------- 多算例 On-policy 核心支持 ---------------------- #
class ALNSTrainingContext:
    def __init__(self, instance_path, args, num_steps):
        self.instance_path = str(instance_path)
        self.num_steps = int(num_steps)

        self.model = Model()
        self.model.rand_d_max = 0.25
        self.model.rand_d_min = 0.1
        self.model.T0 = 50.0

        readCsvFile(self.instance_path, self.model)
        calDistanceMatrix(self.model)

        self.nodes = [self.model.depot] + [self.model.demand_dict[node_id] for node_id in self.model.node_id_list]

        self.base_edge_index, self.base_edge_weight = precompute_base_graph(
            self.nodes,
            self.model,
            k_nn=args.knn_k,
            use_full_graph_knn=args.full_graph_knn,
            weight_mode=args.weight_mode
        )

        self.initial_temperature = 50.0
        self.reset()

    def reset(self):
        self.sol = Sol()
        self.sol.node_no_seq = genInitialSol(self.model)
        self.sol.obj, self.sol.route_list, self.sol.route_distance = calObj(self.sol.node_no_seq, self.model)

        self.model.best_sol = copy.deepcopy(self.sol)
        self.initial_obj = float(self.sol.obj)
        self.model.T0 = self.initial_temperature
        self.model.phi = (0.05 / self.initial_temperature) ** (1.0 / max(1, self.num_steps))

        self.iteration = 0
        self.stagnation_steps = 0
        self.done = False

        self.model.d_select[:] = 0.0
        self.model.r_select[:] = 0.0

def collect_one_on_policy_transition(context, agent, rollout_buffer):
    model = context.model
    sol = context.sol
    nodes = context.nodes

    it = context.iteration
    num_steps = context.num_steps

    progress = it / max(1, num_steps - 1)
    temperature_ratio = model.T0 / max(context.initial_temperature, 1e-6)
    best_gap = (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6)
    stagnation_ratio = context.stagnation_steps / max(1, num_steps)

    g_full = build_graph(
        sol,
        nodes,
        model,
        context.base_edge_index,
        context.base_edge_weight,
        removed_set=None,
        include_path_edges=True
    )

    full_state = make_state_record(
        graph=g_full,
        stage=0,
        destroy_ratio=0.0,
        destroy_id=-1,
        progress=progress,
        temperature_ratio=temperature_ratio,
        best_gap=best_gap,
        stagnation_ratio=stagnation_ratio
    )

    with torch.no_grad():
        full_tensor = encode_single_state(agent, full_state)
        destroy_action, destroy_probs = agent.act_destroy(full_tensor, deterministic=False)

    model.d_select[destroy_action] += 1

    destroyed_routes,removed = doDestroy_by_id(destroy_action, model, sol, it, num_steps)
    removed_set = set(removed)
    destroy_ratio = len(removed) / max(1, len(model.node_id_list))

    partial_sol = apply_destroy_to_partial_sol_explicit(destroyed_routes, model)

    g_partial = build_graph(
        partial_sol,
        nodes,
        model,
        context.base_edge_index,
        context.base_edge_weight,
        removed_set=removed_set,
        include_path_edges=True
    )

    partial_state = make_state_record(
        graph=g_partial,
        stage=1,
        destroy_ratio=destroy_ratio,
        destroy_id=destroy_action,
        progress=progress,
        temperature_ratio=temperature_ratio,
        best_gap=best_gap,
        stagnation_ratio=stagnation_ratio
    )

    with torch.no_grad():
        partial_tensor = encode_single_state(agent, partial_state)
        repair_action, repair_probs = agent.act_repair(partial_tensor, deterministic=False)

    model.r_select[repair_action] += 1

    new_sol = doRepair_by_id(repair_action,destroyed_routes, removed, model, sol)

    current_obj_before = float(sol.obj)
    best_obj_before = float(model.best_sol.obj)
    candidate_obj = float(new_sol.obj)

    # 先做接受判断
    if candidate_obj < current_obj_before:
        accepted = True
    else:
        acceptance_probability = math.exp(
            (current_obj_before - candidate_obj)
            / max(model.T0, 1e-6)
        )
        accepted = acceptance_probability >= random.random()

    if accepted:
        context.sol = copy.deepcopy(new_sol)
        sol = context.sol

    # 更新最优
    if candidate_obj < model.best_sol.obj:
        model.best_sol = copy.deepcopy(new_sol)
        found_new_best = True
        context.stagnation_steps = 0
    else:
        found_new_best = False
        context.stagnation_steps += 1

    next_obj = float(sol.obj)
    best_obj_after = float(model.best_sol.obj)
    destroy_reward = 0.0
    repair_reward = calculate_transition_reward(
        current_obj=current_obj_before,
        candidate_obj=candidate_obj,
        best_obj_before=best_obj_before,
        best_obj_after=best_obj_after,
        initial_obj=context.initial_obj,
        accepted=accepted
    )
    # 1. 步数与终止标记更新
    context.iteration += 1
    done = float(context.iteration >= num_steps)
    context.done = bool(done)

    # 2. 温度指数冷却
    model.T0 *= model.phi

    # 3. 构造实际下一状态的图（若拒绝，则直接复用当前图拓扑以节省计算开销）
    if accepted:
        next_graph = build_graph(
            context.sol,
            nodes,
            model,
            context.base_edge_index,
            context.base_edge_weight,
            removed_set=None,
            include_path_edges=True
        )
    else:
        next_graph = g_full

    # 4. 只计算一次下一状态的各项比例特征（结合边界显式裁剪）
    next_progress = float(np.clip(
        context.iteration / max(1, num_steps - 1),
        0.0,
        1.0
    ))

    next_temperature_ratio = (
            model.T0 / max(context.initial_temperature, 1e-6)
    )

    next_best_gap = (
                            context.sol.obj - model.best_sol.obj
                    ) / max(abs(model.best_sol.obj), 1e-6)

    next_stagnation_ratio = (
            context.stagnation_steps / max(1, num_steps)
    )

    # 5. 构造下一完整状态记录
    next_full_state = make_state_record(
        graph=next_graph,
        stage=0,
        destroy_ratio=0.0,
        destroy_id=-1,
        progress=next_progress,
        temperature_ratio=next_temperature_ratio,
        best_gap=next_best_gap,
        stagnation_ratio=next_stagnation_ratio
    )

    # 6. 存入 on-policy 缓冲区
    rollout_buffer.add(
        full_state=full_state,
        destroy_action=destroy_action,
        partial_state=partial_state,
        repair_action=repair_action,
        repair_reward=repair_reward,
        next_full_state=next_full_state,
        done=done
    )

    return {
        "destroy_probs": destroy_probs,
        "repair_probs": repair_probs,
        "repair_reward": repair_reward,
        "accepted": accepted,
        "best_obj": model.best_sol.obj
    }

def train_multi_instance_on_policy(agent, instance_paths, args):
    rollout_buffer = RolloutBuffer()
    contexts = []

    for instance_path in instance_paths:
        temporary_model = Model()
        readCsvFile(instance_path, temporary_model)

        instance_steps = max(
            100,
            max(
                50 * temporary_model.number_of_demands,
                2000
            ) // 10
        )

        context = ALNSTrainingContext(
            instance_path=instance_path,
            args=args,
            num_steps=instance_steps
        )
        contexts.append(context)

    total_environment_steps = (
        args.train_epochs
        * sum(context.num_steps for context in contexts)
    )

    global_environment_step = 0
    update_count = 0

    for epoch in range(args.train_epochs):
        random.shuffle(contexts)
        for context in contexts:
            context.reset()

        while True:
            active_contexts = [context for context in contexts if not context.done]
            if not active_contexts:
                break

            random.shuffle(active_contexts)

            for context in active_contexts:
                collect_one_on_policy_transition(
                    context=context,
                    agent=agent,
                    rollout_buffer=rollout_buffer
                )

                global_environment_step += 1

                if len(rollout_buffer) >= args.rollout_size:
                    global_progress = global_environment_step / max(1, total_environment_steps)
                    entropy_coef = get_entropy_coef(global_progress, entropy_start=0.05, entropy_end=0.02)

                    statistics = agent.update_rollout(
                        rollout_buffer,
                        entropy_coef=entropy_coef
                    )

                    update_count += 1

                    # if update_count % 10 == 0 and statistics is not None:
                    #     print(
                    #         f"[Epoch {epoch + 1}/{args.train_epochs}] "
                    #         f"Update={update_count} | "
                    #         f"Loss={statistics['total_loss']:.4f} | "
                    #         f"Critic={statistics['critic_loss']:.4f} | "
                    #         f"Entropy D/R={statistics['destroy_entropy']:.4f}/{statistics['repair_entropy']:.4f} | "
                    #         f"TD D/R={statistics['destroy_td_abs']:.4f}/{statistics['repair_td_abs']:.4f}"
                    #     )

        if len(rollout_buffer) > 1:
            global_progress = global_environment_step / max(1, total_environment_steps)
            entropy_coef = get_entropy_coef(global_progress, entropy_start=0.05, entropy_end=0.02)
            agent.update_rollout(rollout_buffer, entropy_coef=entropy_coef)

        epoch_results = [
            (Path(context.instance_path).name, context.model.best_sol.obj)
            for context in contexts
        ]
        print(f"[Epoch {epoch + 1}] 训练实例结果: {epoch_results}")
    return global_environment_step
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
    parser = argparse.ArgumentParser(description='DAC-ALNS Unified Single Critic')
    parser.add_argument('--instance_path', type=str, required=True, help='单个实例(CSV)文件路径')
    parser.add_argument('--result_path', type=str, required=True, help='结果文件路径')
    parser.add_argument('--knn_k', type=int, default=5, help='kNN邻居数')
    parser.add_argument('--full_graph_knn', action='store_true', help='使用全连接图替代kNN')
    parser.add_argument('--weight_mode', choices=['inv', 'dist'], default='inv', help='边权计算模式')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--eps', type=float, default=0.10, help='兼容旧逻辑的epsilon（实际上不使用）')
    parser.add_argument('--mode', type=str, choices=['train', 'solve'], default='train', help='模式')
    parser.add_argument('--model_path', type=str, default=None, help='模型保存/读取路径')

    # 多算例 CLI 补充参数
    parser.add_argument('--train_instance_dir', type=str, default=None, help='多算例训练目录')
    parser.add_argument('--train_epochs', type=int, default=10, help='多算例训练轮次')
    parser.add_argument('--rollout_size', type=int, default=64, help='联合 rollout 更新更新批次大小')

    args = parser.parse_args()
    set_seed(args.seed)

    gamma = 0.99

    # 初始化单实例相关的环境
    model = Model()
    readCsvFile(args.instance_path, model)
    calDistanceMatrix(model)

    nodes = [model.depot] + [model.demand_dict[nid] for nid in model.node_id_list]

    sol = Sol()
    sol.node_no_seq = genInitialSol(model)

    # 【修改】：首次运行通过 DP Split 建立正确的显式路线格式，之后整个流程全部使用此显式路线
    sol.route_list, total_cost = splitRoutes(sol.node_no_seq, model)
    sol.obj = total_cost

    route_distances = []
    for route in sol.route_list:
        idx = [model.id2idx[node_id] for node_id in route]
        d = model.distance_matrix[idx[:-1], idx[1:]].sum()
        route_distances.append(float(d))
    sol.route_distance = route_distances

    model.best_sol = copy.deepcopy(sol)

    base_edge_index, base_edge_weight = precompute_base_graph(
        nodes, model, k_nn=args.knn_k, use_full_graph_knn=args.full_graph_knn, weight_mode=args.weight_mode
    )

    agent = DACAgent(
        enc_dim=64,
        hid=64,
        gamma_repair=gamma,
        critic_coef=1.0
    ).to(device)
    resume_global_step = 0

    if args.model_path and os.path.exists(args.model_path):
        resume_global_step = load_training_checkpoint(
            agent,
            args.model_path,
            load_optimizers=(args.mode == "train")
        )
        print(
            f"已加载 checkpoint: {args.model_path}, "
            f"global_step={resume_global_step}"
        )
    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"加载已有模型继续训练/求解: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)

        if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["agent_state_dict"])
            resume_global_step = int(checkpoint.get("global_step", 0))
        else:
            # 兼容旧版只保存 agent.state_dict() 的模型
            agent.load_state_dict(checkpoint)
    # ---------------------- [重要] 12. 多算例训练逻辑触发 ---------------------- #
    if args.mode == "train" and args.train_instance_dir is not None:
        train_directory = Path(args.train_instance_dir)
        instance_paths = sorted(str(path) for path in train_directory.glob("*.csv"))

        if not instance_paths:
            raise ValueError(f"训练目录中没有 CSV 文件: {train_directory}")

        print(f"使用 {len(instance_paths)} 个算例进行多算例 on-policy DUEL-AC 训练。")

        global_environment_step = train_multi_instance_on_policy(
            agent=agent,
            instance_paths=instance_paths,
            args=args
        )

        if args.model_path:
            model_save_path = Path(args.model_path)
        else:
            model_save_path = Path(args.result_path).parent / "duel_ac_multi_instance.pth"

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_training_checkpoint(
            agent,
            model_save_path,
            global_step=global_environment_step
        )
        print(f"多算例模型已保存至: {model_save_path}")
        raise SystemExit(0)

    # ---------------------- 传统/单算例主流程 ---------------------- #
    if args.mode == "solve":
        if args.model_path is None or not os.path.exists(args.model_path):
            raise ValueError("solve 模式下必须提供 --model_path。")
        agent.eval()
    else:
        agent.train()

    num_steps = max(50 * model.number_of_demands, 2000)
    if args.mode == 'train':
        num_steps = max(100, num_steps // 4)
        print(f"训练模式：ALNS 迭代次数缩减至: {num_steps}")
    else:
        print(f"求解模式：全量步数迭代: {num_steps}")

    rollout_buffer = RolloutBuffer()
    rollout_size = 32
    model.T0 = 50
    initial_temperature = float(model.T0)
    stagnation_steps = 0
    initial_obj = float(sol.obj)

    recent_d_probs = []
    recent_r_probs = []
    window_size = 200

    history_best_obj = [sol.obj]
    return_list = []
    start_time = time.time()
    all_iterations = []

    model.phi = (0.05 / initial_temperature) ** (1.0 / max(1, num_steps))

    t_graph_build = 0.0
    t_inference = 0.0
    t_training = 0.0

    # ---------------------- 单算例主循环 (完全重构) ---------------------- #
    for it in range(num_steps):
        iteration_progress = it / max(1, num_steps - 1)
        temperature_ratio = model.T0 / max(initial_temperature, 1e-6)
        current_best_gap = (sol.obj - model.best_sol.obj) / max(abs(model.best_sol.obj), 1e-6)
        stagnation_ratio = stagnation_steps / max(1, num_steps)

        # =====================================================
        # 1. 当前完整状态
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
        # 2. Destroy Actor 选择算子
        # =====================================================
        t0 = time.perf_counter()
        deterministic = False

        a_d, d_prob = agent.act_destroy(
            full_state_tensor,
            deterministic=deterministic
        )

        t_inference += (time.perf_counter() - t0)
        model.d_select[a_d] += 1

        destroyed_routes, remove_list = doDestroy_by_id(a_d, model, sol, it, num_steps)
        removed_set = set(remove_list)
        destroy_ratio = len(remove_list) / max(1, len(model.node_id_list))

        # =====================================================
        # 3. 部分状态
        # =====================================================
        t0 = time.perf_counter()

        part_sol = apply_destroy_to_partial_sol_explicit(destroyed_routes, model)
        g_partial = build_graph(
            part_sol,
            nodes,
            model,
            base_edge_index,
            base_edge_weight,
            removed_set=removed_set,
            include_path_edges=True
        )

        partial_state_record = make_state_record(
            graph=g_partial,
            stage=1,
            destroy_ratio=destroy_ratio,
            destroy_id=a_d,
            progress=iteration_progress,
            temperature_ratio=temperature_ratio,
            best_gap=current_best_gap,
            stagnation_ratio=stagnation_ratio
        )

        with torch.no_grad():
            partial_state_tensor = encode_single_state(agent, partial_state_record)

        t_graph_build += (time.perf_counter() - t0)


        # =====================================================
        # 4. Repair Actor 选择算子
        # =====================================================
        t0 = time.perf_counter()

        a_r, r_prob = agent.act_repair(
            partial_state_tensor,
            deterministic=deterministic
        )

        t_inference += (time.perf_counter() - t0)
        model.r_select[a_r] += 1

        new_sol = doRepair_by_id(a_r, destroyed_routes, remove_list, model)

        # =====================================================
        # 5. 完整候选解奖励
        # =====================================================
        current_obj_before = float(sol.obj)
        best_obj_before = float(model.best_sol.obj)
        candidate_obj = float(new_sol.obj)

        # 先做接受判断
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

        # 更新最优
        if candidate_obj < model.best_sol.obj:
            model.best_sol = copy.deepcopy(new_sol)
            found_new_best = True
            stagnation_steps = 0
        else:
            found_new_best = False
            stagnation_steps += 1

        next_obj = float(sol.obj)
        best_obj_after = float(model.best_sol.obj)
        destroy_reward = 0.0

        repair_reward = calculate_transition_reward(
            current_obj=current_obj_before,
            candidate_obj=candidate_obj,
            best_obj_before=best_obj_before,
            best_obj_after=best_obj_after,
            initial_obj=initial_obj,
            accepted=accepted
        )

        # =====================================================
        # 7. 实际下一完整状态
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
        # 8. 保存 on-policy rollout
        # =====================================================
        if args.mode == "train":
            rollout_buffer.add(
                full_state=full_state_record,
                destroy_action=a_d,
                partial_state=partial_state_record,
                repair_action=a_r,
                repair_reward=repair_reward,
                next_full_state=next_full_state_record,
                done=is_terminal
            )

        return_list.append(destroy_reward)
        return_list.append(repair_reward)

        recent_d_probs.append(d_prob)
        recent_r_probs.append(r_prob)

        if len(recent_d_probs) > window_size:
            recent_d_probs = recent_d_probs[-window_size:]
        if len(recent_r_probs) > window_size:
            recent_r_probs = recent_r_probs[-window_size:]

        # =====================================================
        # 9. Rollout 触发参数更新
        # =====================================================
        should_update = (
            args.mode == "train"
            and (len(rollout_buffer) >= rollout_size or is_terminal > 0.5)
        )

        if should_update:
            t0 = time.perf_counter()
            entropy_coef = get_entropy_coef(
                global_progress=iteration_progress,
                entropy_start=0.05,
                entropy_end=0.02
            )

            update_statistics = agent.update_rollout(
                rollout_buffer,
                entropy_coef=entropy_coef
            )
            t_training += (time.perf_counter() - t0)

            if update_statistics is not None:
                print(
                    "[DUEL-AC Update] "
                    f"Loss={update_statistics['total_loss']:.4f} | "
                    f"Critic={update_statistics['critic_loss']:.4f} | "
                    f"Entropy D/R={update_statistics['destroy_entropy']:.4f}/{update_statistics['repair_entropy']:.4f} | "
                    f"TD D/R={update_statistics['destroy_td_abs']:.4f}/{update_statistics['repair_td_abs']:.4f} | "
                    f"Grad Enc/Critic/D/R="
                    f"{update_statistics['encoder_grad_norm']:.4f}/"
                    f"{update_statistics['critic_grad_norm']:.4f}/"
                    f"{update_statistics['destroy_grad_norm']:.4f}/"
                    f"{update_statistics['repair_grad_norm']:.4f}"
                )

        history_best_obj.append(model.best_sol.obj)

        # ---------------------- 打印与记录 ---------------------- #
        if (it + 1) % 200 == 0:
            avg_d_prob = np.mean(recent_d_probs, axis=0)
            avg_r_prob = np.mean(recent_r_probs, axis=0)
            d_prob_str = ", ".join(f"{probability:.3f}" for probability in avg_d_prob)
            r_prob_str = ", ".join(f"{probability:.3f}" for probability in avg_r_prob)

            # print(
            #     f"{Path(args.instance_path).name} | "
            #     f"Iter {it + 1}/{num_steps} | "
            #     f"Best Obj: {model.best_sol.obj:.2f} | "
            #     f"Current Obj: {sol.obj:.2f} | "
            #     f"Time: {time.time() - start_time:.2f}s"
            # )
            # print(f"    [Rolling Monitor] Avg Destroy Probs: [{d_prob_str}]")
            # print(f"    [Rolling Monitor] Avg Repair Probs:  [{r_prob_str}]")

        if (it + 1) % 10 == 0:
            all_iterations.append({
                "iteration": it + 1,
                "file_path": args.instance_path,
                "best_obj": round(model.best_sol.obj, 2),
                "current_obj": round(sol.obj, 2),
                "repair_reward": round(repair_reward, 6),
                "accepted": bool(accepted),
                "new_best": bool(found_new_best),
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
        "Iterations": num_steps,
        "Device": str(device)
    }

    # ---------------------- 收尾与文件输出 ---------------------- #
    parent_path = Path(args.result_path).parent
    parts = Path(args.result_path).parts
    file_name = Path(args.result_path).stem

    csv_summary_path = parent_path / "DAC_ALNS_Summary_Results.csv"
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

    if args.mode == 'train':
        if args.model_path:
            model_save_path = Path(args.model_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(
                agent,
                model_save_path,
                global_step=num_steps
            )
            print(f"模型已保存至: {model_save_path}")
        else:
            model_save_path = parent_path / f'{file_name}_model_{round(model.best_sol.obj, 2)}.pth'
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(
                agent,
                model_save_path,
                global_step=num_steps
            )
            print(f"模型已自动保存至: {model_save_path}")

    print("\n--- Final Statistics ---")
    print(f"Best Objective Found: {model.best_sol.obj:.2f}")
    print(f"Total Run Time: {run_time:.2f}s")
    print(f"Neural Network Overhead Ratio: {nn_percentage:.2f}%")
    print(f"Destroy Op 1 Selections (Random):  {model.d_select[0]}")
    print(f"Destroy Op 2 Selections (Worst):   {model.d_select[1]}")
    print(f"Destroy Op 3 Selections (Shaw):    {model.d_select[2]}")
    print(f"Destroy Op 4 Selections (Seq):     {model.d_select[3]}")
    print(f"Destroy Op 5 Selections (Route):   {model.d_select[4]}")
    print(f"Repair  Op 1 Selections (RndBest): {model.r_select[0]}")
    print(f"Repair  Op 2 Selections (Regret):  {model.r_select[1]}")
    print(f"Repair  Op 3 Selections (Cheap):   {model.r_select[2]}")
    print(f"Repair  Op 4 Selections (Noise):   {model.r_select[3]}")
    print(f"Repair  Op 5 Selections (Granu):   {model.r_select[4]}")
    print("------------------------\n")