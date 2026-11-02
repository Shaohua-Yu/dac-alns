# -*- coding: utf-8 -*-
"""Reijnen-style DRL-PPO online baseline for the local CVRP framework.

The CVRP environment, initial solution, five destroy operators, five repair
operators, capacity checks, and route evaluation are embedded in this file
from the control-aligned ALNS implementation. Method-specific decisions follow the
public Reijnen et al. implementation as closely as the online protocol allows:

* the paper's seven-dimensional search-statistics state and an MLP policy;
* four simultaneous categorical actions for destroy, repair, destruction
  severity, and simulated-annealing temperature;
* a sparse reward of 5 only for a new global best, and 0 otherwise;
* clipped PPO updates over the summed log-probability of the four actions.
* a 20% uniform exploration mixture that protects every action probability.

The online comparison uses one eight-iteration rollout per update event.  The
ALNS environment and iteration budget are aligned with DAC, while PPO retains
its own four-action control, reward, state representation, and loss.

This is an adapted online baseline, not an unchanged reproduction of the
authors' train-then-deploy DR-ALNS implementation.
"""

import argparse
import csv
import os
import copy
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------- 内置 CVRP 环境与 5x5 算子 ---------------------- #
# 与 ALNS_CVRP_5x5_Control_Aligned.py 使用同一实现，但本文件可独立运行。
# ---------------------- 结果追加与保存 (带文件锁的安全版本) ---------------------- #
def safe_append_to_json(file_path, problem_name, obj, timestamp):
    """
    带有文件锁和原子写入的安全 JSON 追加函数，专为多进程并发设计。
    """
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

        problem_data = data.setdefault(problem_name, {
            "objectives": [],
            "times": []
        })
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
    """将运行结果追加到同一 CSV 表格中"""
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=record_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record_dict)


# ---------------------- 基础数据结构 ---------------------- #
class Sol():
    def __init__(self):
        self.node_no_seq = None
        self.obj = None
        self.route_list = None
        self.route_distance = None


class Node():
    def __init__(self):
        self.id = 0
        self.x_coord = 0
        self.y_coord = 0
        self.demand = 0


class Model():
    def __init__(self):
        self.best_sol = None
        self.node_id_list = []
        self.demand_dict = {}
        self.depot = None
        self.number_of_demands = 0
        self.vehicle_cap = 100
        self.distance_matrix = {}
        self.rand_d_max = 0.25
        self.rand_d_min = 0.1
        self.T0 = 50.0
        self.phi = 0.995
        self.rho = 0.1
        self.max_repair_id = 5  # 5个修复算子

        # 5个Destroy * 5个Repair = 25种组合
        self.dr_weight = np.ones(25)
        self.dr_score = np.zeros(25)
        self.dr_history_select = np.zeros(25)
        self.dr_history_score = np.zeros(25)


def readCsvFile(demand_file, model):
    with open(demand_file, 'r', encoding='utf-8') as f:
        demand_reader = csv.DictReader(f)
        v_cap_assigned = False
        for row in demand_reader:
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
        model.number_of_demands = len(model.node_id_list)


# ---------------------- 距离计算与分段 ---------------------- #
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

    # 1. 额外初始化需求快速查询数组（计算相似度所需）
    max_id = max(model.demand_dict.keys())
    model.demands_arr = np.zeros(max_id + 1, dtype=np.float32)
    for nid, node in model.demand_dict.items():
        model.demands_arr[nid] = node.demand

    # 2. 预计算 Shaw 相似度矩阵
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


def genInitialSol(model):
    """与 DAC-ALNS 一致的最近邻初解。"""
    unvisited = copy.deepcopy(model.node_id_list)
    if not unvisited:
        return []

    seq = []
    current = model.depot.id
    while unvisited:
        nearest = min(
            unvisited,
            key=lambda node_id: model.distance_matrix[
                model.id2idx[current], model.id2idx[node_id]
            ]
        )
        seq.append(nearest)
        unvisited.remove(nearest)
        current = nearest
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
        load = 0
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
                cost = cost - dist_mat[prev_idx, depot_idx] + dist_mat[prev_idx, cur_idx] + dist_mat[cur_idx, depot_idx]

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
    dem_arr = np.array(
        [model.demand_dict[id_].demand for id_ in node_no_seq],
        dtype=np.float32
    )

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
    return sum(model.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))


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


# ==================== 5x5 算子库（完全一致与性能优化版） ==================== #

# --- 1. R1: 随机客户移除 (Random removal) ---
def destroy_strategy_op1(routes, k, dist_mat, demands, capacity):
    all_customers = [c for r in routes for c in r[1:-1]]
    if not all_customers or k <= 0:
        return [r[:] for r in routes], []
    k = min(k, len(all_customers))
    removed = random.sample(all_customers, k)
    removed_set = set(removed)
    new_routes = [[c for c in r if c not in removed_set] for r in routes]
    return [r for r in new_routes if len(r) > 2], sorted(removed)


# --- 2. R16: 最差成本客户移除 (Worst cost removal - One-shot 模式) ---
def destroy_strategy_op2(routes, k, dist_mat, demands, capacity, p_worst=5):
    all_custs = [c for r in routes for c in r[1:-1]]
    if not all_custs or k <= 0:
        return [r[:] for r in routes], []
    k = min(k, len(all_custs))

    # 一次性计算所有节点的节省值（One-shot），降本显著
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


# --- 3. R13: 后验 Shaw 关系移除 (A posteriori score related removal - 预计算矩阵查表) ---
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
            # 运行时执行简单的 O(1) 矩阵查表，符合速度评估
            score = similarity_matrix[ref][c]
            scores.append((score, c))
        scores.sort(key=lambda x: x[0])
        idx = int(math.floor(len(scores) * (random.random() ** p_random)))
        idx = max(0, min(idx, len(scores) - 1))
        removed.append(scores[idx][1])

    rem_set = set(removed)
    new_routes = [[c for c in r if c not in rem_set] for r in routes]
    return [r for r in new_routes if len(r) > 2], sorted(removed)


# --- 4. R9/R10: 邻近序列/字符串移除 (Sequence-based/Cluster Removal - 跨路径支持) ---
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

    # 循环截取：若单条路径不满足提取规模，继续在其他随机路径中截取
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


# 5. R19: 随机多路径整条/块移除 (All customers from random route - 严格服从全局参数 k)
def destroy_strategy_op5(routes, k, dist_mat, demands, capacity):
    active_route_indices = [i for i, r in enumerate(routes) if len(r) > 2]
    if not active_route_indices or k <= 0:
        return [r[:] for r in routes], []

    # 随机打乱活跃路径的遍历顺序
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
            # 1. 路径规模小于等于所需移除量：整条清空（保留R19完全解散路径的特性）
            removed.extend(nodes_in_route)
            new_routes[r_idx] = [0, 0]
        else:
            # 2. 路径规模大于所需移除量：截断提取前 needed 个节点（保留路径块剥离特性）
            to_remove = nodes_in_route[:needed]
            removed.extend(to_remove)

            # 从原路线中过滤掉被移出的节点块，保留其余节点
            rem_set = set(to_remove)
            new_routes[r_idx] = [0] + [c for c in nodes_in_route if c not in rem_set] + [0]

    return [r for r in new_routes if len(r) > 2], sorted(removed)


# --- 6. I5: 随机顺序最佳位置插入 (In random order - at best position) ---
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


# --- 7. I13: 位置后悔值插入 (Highest position regret - Regret-2 经典版) ---
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
                # 恢复经典标准 Regret 计算定义，避免越界优先级偏置
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


# --- 8. I7: 最小成本优先最佳位置插入 (Lowest cost first - at best position) ---
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
                        cost = dist_mat[route[i - 1]][cust] + dist_mat[cust][route[i]] - dist_mat[route[i - 1]][
                            route[i]]
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


# --- 9. I14: 带噪声的位置后悔值插入 (Highest position regret - with noise) ---
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
                        # 引入在最大距离合理比例范围内的随机噪声扰动
                        noise = (random.random() * 2.0 - 1.0) * noise_level * max_dist
                        perturbed_cost = max(0.0, b_cost + noise)
                        insertion_costs.append((perturbed_cost, r_idx, b_pos))

            insertion_costs.sort(key=lambda x: x[0])
            if len(insertion_costs) >= 2:
                regret = insertion_costs[1][0] - insertion_costs[0][0]
            elif len(insertion_costs) == 1:
                # 恢复经典标准 Regret 定义
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


# --- 10. I10: 限制近邻/随机路径插入 (Restricted best position insertion - 限流优化) ---
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

        # 查找该客户在图中的最近邻节点
        neighbors = sorted(
            [(dist_mat[cust][other], other) for other in range(num_nodes) if other != cust and other != 0],
            key=lambda x: x[0])
        nearest_neighbors = {other for _, other in neighbors[:k_neighbors]}

        # 计算并筛选包含最近邻居最多的车辆路径
        candidate_route_indices = []
        for r_idx, route in enumerate(new_routes):
            overlap = sum(1 for node in route if node in nearest_neighbors)
            if overlap > 0:
                candidate_route_indices.append((overlap, r_idx))

        # 按重叠节点数降序排列，严格限制搜索路径的最大上限为有限条路径 (如 limit_routes=3)
        candidate_route_indices.sort(key=lambda x: x[0], reverse=True)
        selected_route_indices = [r_idx for _, r_idx in candidate_route_indices[:limit_routes]]

        # 无近邻路径符合时，随机选取少数几条路径进行候选
        if not selected_route_indices:
            all_indices = list(range(len(new_routes)))
            selected_route_indices = random.sample(all_indices, min(limit_routes, len(all_indices)))

        best_cost = float('inf')
        best_r_idx = -1
        best_pos = -1

        # 仅对受限的目标路径子集进行插入代价评估，显著降低计算开销
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



# 注册算子映射列表
DESTROY_FUNCS = [
    destroy_strategy_op1,
    destroy_strategy_op2,
    destroy_strategy_op3,
    destroy_strategy_op4,
    destroy_strategy_op5
]

REPAIR_FUNCS = [
    repair_strategy_op1,
    repair_strategy_op2,
    repair_strategy_op3,
    repair_strategy_op4,
    repair_strategy_op5
]

# PPO 使用固定破坏比例，因此仅需要共享修复封装。
def doRepair(repair_id, new_routes, removed, model, sol):
    demands_arr = model.demands_arr

    final_routes = REPAIR_FUNCS[repair_id](
        new_routes, removed, model.distance_matrix, demands_arr, model.vehicle_cap
    )

    new_sol = Sol()
    new_sol.route_list = [route for route in final_routes if len(route) > 2]

    route_distances = []
    for route in new_sol.route_list:
        indices = [model.id2idx[node_id] for node_id in route]
        distance = model.distance_matrix[indices[:-1], indices[1:]].sum()
        route_distances.append(float(distance))

    new_sol.route_distance = route_distances
    new_sol.obj = sum(route_distances)
    new_sol.node_no_seq = [
        customer
        for route in new_sol.route_list
        for customer in route[1:-1]
    ]
    return new_sol


torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 7
ACTION_DIMS = (5, 5, 10, 50)
NEW_BEST_REWARD = 5.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_search_state(
        best_improved,
        current_accepted,
        current_improved,
        is_current_best,
        current_obj,
        best_obj,
        stagnation_count,
        iteration,
        max_iterations
):
    """Build the seven problem-agnostic search features from the paper."""
    cost_difference_from_best = (
        (float(current_obj) - float(best_obj))
        / max(abs(float(best_obj)), 1e-9)
        * 100.0
    )
    progress = float(iteration) / max(1, int(max_iterations))
    return np.asarray(
        [
            float(bool(best_improved)),
            float(bool(current_accepted)),
            float(bool(current_improved)),
            float(bool(is_current_best)),
            cost_difference_from_best,
            float(stagnation_count),
            progress,
        ],
        dtype=np.float32,
    )


def destruction_ratio_from_action(action):
    """Map actions 0..9 to destruction ratios 0.1..1.0."""
    return (int(action) + 1) / 10.0


def temperature_from_action(action):
    """Map the paper's actions 0..49 to temperatures 0.1..5.0."""
    return 0.1 * (int(action) + 1)


def destroy_with_ratio(destroy_id, ratio, model, sol):
    """Run a local destroy operator using PPO-selected severity."""
    customer_count = len(model.node_id_list)
    if customer_count <= 1:
        return copy.deepcopy(sol.route_list), []

    remove_count = round(float(ratio) * customer_count)
    remove_count = min(max(1, remove_count), customer_count - 1)
    routes = sol.route_list if sol.route_list else [[0] + sol.node_no_seq + [0]]

    if int(destroy_id) == 2:
        return DESTROY_FUNCS[int(destroy_id)](
            routes,
            remove_count,
            model.distance_matrix,
            model.demands_arr,
            model.vehicle_cap,
            similarity_matrix=model.similarity_matrix,
        )
    return DESTROY_FUNCS[int(destroy_id)](
        routes,
        remove_count,
        model.distance_matrix,
        model.demands_arr,
        model.vehicle_cap,
    )


class RolloutBuffer:
    def __init__(self):
        self.records = []

    def add(self, state, actions, old_log_prob, reward, next_state, done):
        self.records.append(
            {
                "state": np.asarray(state, dtype=np.float32).copy(),
                "actions": np.asarray(actions, dtype=np.int64).copy(),
                "old_log_prob": float(old_log_prob),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32).copy(),
                "done": float(done),
            }
        )

    def clear(self):
        self.records.clear()

    def __len__(self):
        return len(self.records)

    def as_tensors(self):
        if not self.records:
            raise ValueError("RolloutBuffer is empty.")
        return {
            "states": torch.as_tensor(
                np.stack([x["state"] for x in self.records]),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "actions": torch.as_tensor(
                np.stack([x["actions"] for x in self.records]),
                dtype=torch.long,
                device=DEVICE,
            ),
            "old_log_probs": torch.as_tensor(
                [x["old_log_prob"] for x in self.records],
                dtype=torch.float32,
                device=DEVICE,
            ),
            "rewards": torch.as_tensor(
                [x["reward"] for x in self.records],
                dtype=torch.float32,
                device=DEVICE,
            ),
            "next_states": torch.as_tensor(
                np.stack([x["next_state"] for x in self.records]),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "dones": torch.as_tensor(
                [x["done"] for x in self.records],
                dtype=torch.float32,
                device=DEVICE,
            ),
        }


class ReijnenStylePPO(torch.nn.Module):
    """Two-layer MLP PPO with four independent categorical action heads."""

    def __init__(
            self,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ppo_epochs=10,
            minibatch_size=8,
            entropy_coef=0.0,
            exploration_mix=0.2,
            value_coef=0.5,
            max_grad_norm=0.5
    ):
        super().__init__()
        self.policy_mlp = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )
        self.action_heads = torch.nn.ModuleList(
            [torch.nn.Linear(64, action_dim) for action_dim in ACTION_DIMS]
        )
        self.value_mlp = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )

        policy_linears = [
            layer for layer in self.policy_mlp if isinstance(layer, torch.nn.Linear)
        ]
        value_linears = [
            layer for layer in self.value_mlp if isinstance(layer, torch.nn.Linear)
        ]
        for layer in policy_linears:
            torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            torch.nn.init.zeros_(layer.bias)
        for layer in value_linears[:-1]:
            torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(value_linears[-1].weight, gain=1.0)
        torch.nn.init.zeros_(value_linears[-1].bias)
        for head in self.action_heads:
            torch.nn.init.orthogonal_(head.weight, gain=0.01)
            torch.nn.init.zeros_(head.bias)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.entropy_coef = float(entropy_coef)
        self.exploration_mix = float(exploration_mix)
        if not 0.0 <= self.exploration_mix < 1.0:
            raise ValueError("exploration_mix must be in [0, 1)")
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

    def distributions(self, states):
        policy_features = self.policy_mlp(states)
        distributions = []
        for head in self.action_heads:
            raw_probabilities = F.softmax(head(policy_features), dim=-1)
            action_count = raw_probabilities.size(-1)
            probabilities = (
                (1.0 - self.exploration_mix) * raw_probabilities
                + self.exploration_mix / action_count
            )
            distributions.append(
                torch.distributions.Categorical(probs=probabilities)
            )
        return distributions

    def value(self, states):
        return self.value_mlp(states).squeeze(-1)

    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).view(1, -1)
        distributions = self.distributions(state_tensor)

        if deterministic:
            action_tensors = [dist.probs.argmax(dim=-1) for dist in distributions]
        else:
            action_tensors = [dist.sample() for dist in distributions]

        log_prob = sum(
            dist.log_prob(action)
            for dist, action in zip(distributions, action_tensors)
        )
        actions = np.asarray(
            [int(action.item()) for action in action_tensors],
            dtype=np.int64,
        )
        probabilities = [
            dist.probs.detach().cpu().squeeze(0).numpy()
            for dist in distributions
        ]
        return actions, float(log_prob.item()), probabilities

    def evaluate_actions(self, states, actions):
        distributions = self.distributions(states)
        log_prob = torch.zeros(states.size(0), device=states.device)
        entropy = torch.zeros(states.size(0), device=states.device)
        for head_id, dist in enumerate(distributions):
            log_prob = log_prob + dist.log_prob(actions[:, head_id])
            entropy = entropy + dist.entropy()
        return log_prob, entropy, self.value(states)

    @staticmethod
    def normalize_advantages(advantages):
        if advantages.numel() <= 1:
            return advantages
        return (
            advantages - advantages.mean()
        ) / advantages.std(unbiased=False).clamp_min(1e-8)

    def update(self, rollout_buffer):
        batch = rollout_buffer.as_tensors()
        with torch.no_grad():
            values = self.value(batch["states"])
            next_values = self.value(batch["next_states"])
            deltas = (
                batch["rewards"]
                + self.gamma * (1.0 - batch["dones"]) * next_values
                - values
            )
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros((), device=DEVICE)
            for step in reversed(range(len(deltas))):
                non_terminal = 1.0 - batch["dones"][step]
                gae = (
                    deltas[step]
                    + self.gamma * self.gae_lambda * non_terminal * gae
                )
                advantages[step] = gae
            returns = advantages + values

        sample_count = len(batch["states"])
        epoch_stats = []
        for _ in range(self.ppo_epochs):
            shuffled = torch.randperm(sample_count, device=DEVICE)
            for start in range(0, sample_count, self.minibatch_size):
                indices = shuffled[start:start + self.minibatch_size]
                new_log_prob, entropy, new_values = self.evaluate_actions(
                    batch["states"][indices],
                    batch["actions"][indices],
                )
                log_ratio = new_log_prob - batch["old_log_probs"][indices]
                ratio = torch.exp(log_ratio)
                advantages_mb = self.normalize_advantages(advantages[indices])
                unclipped = ratio * advantages_mb
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * advantages_mb

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(new_values, returns[indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_range
                    ).float().mean()
                epoch_stats.append(
                    {
                        "loss": float(loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_mean.item()),
                        "approximate_kl": float(approximate_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "ratio": float(ratio.mean().item()),
                        "grad_norm": float(grad_norm),
                    }
                )

        rollout_buffer.clear()
        statistics = {
            name: float(np.mean([entry[name] for entry in epoch_stats]))
            for name in epoch_stats[0]
        }
        statistics["optimizer_steps"] = len(epoch_stats)
        return statistics


def mean_probabilities(probability_records, head_id):
    return np.mean([record[head_id] for record in probability_records], axis=0)


def initialize_solution(model):
    solution = Sol()
    solution.node_no_seq = genInitialSol(model)
    solution.obj, solution.route_list, solution.route_distance = calObj(
        solution.node_no_seq,
        model,
    )
    return solution


def run(args):
    set_seed(args.seed)
    model = Model()
    readCsvFile(args.instance_path, model)
    calDistanceMatrix(model)

    solution = initialize_solution(model)
    model.best_sol = copy.deepcopy(solution)
    num_steps = (
        int(args.num_steps)
        if args.num_steps is not None
        else max(50 * model.number_of_demands, 2000)
    )
    if num_steps <= 0:
        raise ValueError("--num_steps must be positive.")

    agent = ReijnenStylePPO(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        exploration_mix=args.exploration_mix,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    ).to(DEVICE)
    agent.train()

    rollout_buffer = RolloutBuffer()
    action_counts = [np.zeros(size, dtype=np.int64) for size in ACTION_DIMS]
    recent_probabilities = deque(maxlen=200)
    all_iterations = []
    best_history = [float(solution.obj)]
    reward_history = []

    initial_state = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    state = initial_state
    stagnation_count = 0
    online_updates = 0
    ppo_gradient_steps = 0
    t_state = 0.0
    t_inference = 0.0
    t_training = 0.0
    start_time = time.time()

    # print(
    #     "Reijnen-style DRL-PPO Online: "
    #     f"state=7D MLP, actions=5x5x10x50, steps={num_steps}, "
    #     f"rollout={args.rollout_size}, epochs={args.ppo_epochs}, "
    #     f"exploration_mix={args.exploration_mix:g}."
    # )

    for iteration in range(num_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            actions, old_log_prob, probabilities = agent.act(state)
        t_inference += time.perf_counter() - t0

        destroy_id, repair_id, severity_action, temperature_action = (
            int(x) for x in actions
        )
        destroy_ratio = destruction_ratio_from_action(severity_action)
        temperature = temperature_from_action(temperature_action)
        for head_id, action in enumerate(actions):
            action_counts[head_id][int(action)] += 1
        recent_probabilities.append(probabilities)

        destroyed_routes, removed = destroy_with_ratio(
            destroy_id,
            destroy_ratio,
            model,
            solution,
        )
        candidate = doRepair(
            repair_id,
            destroyed_routes,
            removed,
            model,
            solution,
        )

        current_obj_before = float(solution.obj)
        best_obj_before = float(model.best_sol.obj)
        candidate_obj = float(candidate.obj)
        current_improved = candidate_obj < current_obj_before
        found_new_best = candidate_obj < best_obj_before

        if found_new_best or current_improved:
            accepted = True
        else:
            acceptance_probability = math.exp(
                (current_obj_before - candidate_obj) / max(temperature, 1e-9)
            )
            accepted = random.random() < acceptance_probability

        if accepted:
            solution = copy.deepcopy(candidate)
        if found_new_best:
            model.best_sol = copy.deepcopy(candidate)
            stagnation_count = 0
        else:
            stagnation_count += 1

        reward = NEW_BEST_REWARD if found_new_best else 0.0
        t0 = time.perf_counter()
        next_state = build_search_state(
            best_improved=found_new_best,
            current_accepted=accepted,
            current_improved=current_improved,
            is_current_best=math.isclose(
                float(solution.obj),
                float(model.best_sol.obj),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            current_obj=solution.obj,
            best_obj=model.best_sol.obj,
            stagnation_count=stagnation_count,
            iteration=iteration + 1,
            max_iterations=num_steps,
        )
        t_state += time.perf_counter() - t0

        done = float(iteration == num_steps - 1)
        rollout_buffer.add(
            state,
            actions,
            old_log_prob,
            reward,
            next_state,
            done,
        )
        state = next_state
        reward_history.append(reward)
        best_history.append(float(model.best_sol.obj))

        if len(rollout_buffer) >= args.rollout_size or done > 0.5:
            t0 = time.perf_counter()
            update_stats = agent.update(rollout_buffer)
            t_training += time.perf_counter() - t0
            online_updates += 1
            ppo_gradient_steps += int(update_stats["optimizer_steps"])
            # print(
            #     f"[PPO Update {online_updates}] "
            #     f"loss={update_stats['loss']:.4f}, "
            #     f"KL={update_stats['approximate_kl']:.6f}, "
            #     f"clip={update_stats['clip_fraction']:.3f}, "
            #     f"ratio={update_stats['ratio']:.3f}"
            # )

        if (iteration + 1) % 200 == 0:
            avg_destroy = mean_probabilities(recent_probabilities, 0)
            avg_repair = mean_probabilities(recent_probabilities, 1)
            severity_frequency = action_counts[2] / max(1, action_counts[2].sum())
            temperature_frequency = action_counts[3] / max(1, action_counts[3].sum())
            most_used_temperatures = np.argsort(temperature_frequency)[-5:][::-1]
            # print(
            #     f"{Path(args.instance_path).name} | "
            #     f"Iter {iteration + 1}/{num_steps} | "
            #     f"Best {model.best_sol.obj:.2f} | "
            #     f"Current {solution.obj:.2f} | "
            #     f"Time {time.time() - start_time:.2f}s"
            # )
            # print("  Mean destroy probabilities:", np.round(avg_destroy, 3))
            # print("  Mean repair probabilities: ", np.round(avg_repair, 3))
            # print("  Severity frequencies:      ", np.round(severity_frequency, 3))
            # print(
            #     "  Most used temperatures:    ",
            #     [
            #         (
            #             round(temperature_from_action(action), 4),
            #             round(float(temperature_frequency[action]), 3),
            #         )
            #         for action in most_used_temperatures
            #     ],
            # )

        if (iteration + 1) % 10 == 0:
            all_iterations.append(
                {
                    "iteration": iteration + 1,
                    "file_path": args.instance_path,
                    "best_obj": round(float(model.best_sol.obj), 6),
                    "current_obj": round(float(solution.obj), 6),
                    "candidate_obj": round(candidate_obj, 6),
                    "reward": reward,
                    "accepted": bool(accepted),
                    "new_best": bool(found_new_best),
                    "destroy_action": destroy_id,
                    "repair_action": repair_id,
                    "severity_action": severity_action,
                    "destroy_ratio": destroy_ratio,
                    "removed_customers": len(removed),
                    "temperature_action": temperature_action,
                    "temperature": temperature,
                    "state": [round(float(value), 6) for value in state],
                    "run_time": round(time.time() - start_time, 3),
                }
            )

    run_time = time.time() - start_time
    learning_overhead = t_state + t_inference + t_training
    overhead_ratio = learning_overhead / max(run_time, 1e-9) * 100.0
    parent_path = Path(args.result_path).parent
    parent_path.mkdir(parents=True, exist_ok=True)
    file_name = Path(args.result_path).stem

    summary_record = {
        "Instance": Path(args.instance_path).name,
        "Best_Obj": round(float(model.best_sol.obj), 6),
        "Total_Time(s)": round(run_time, 3),
        "State_Build(s)": round(t_state, 3),
        "MLP_Inference(s)": round(t_inference, 3),
        "PPO_Training(s)": round(t_training, 3),
        "Learning_Overhead(s)": round(learning_overhead, 3),
        "Learning_Overhead_Ratio(%)": round(overhead_ratio, 2),
        "Method": "Reijnen_Style_DRL_PPO_Online_Adapted_5x5_Four_Action_Control",
        "State_Mode": "paper_7D_search_statistics_MLP",
        "Action_Mode": "MultiDiscrete_5_5_10_50",
        "Reward_Mode": "5_if_new_global_best_else_0",
        "Temperature_Mode": "50_levels_0.1_to_5.0",
        "Severity_Mode": "ten_levels_0.1_to_1.0",
        "Training_Mode": "per_instance_online_from_random_initialization",
        "Online_Updates": online_updates,
        "PPO_Gradient_Steps": ppo_gradient_steps,
        "Update_Interval": args.rollout_size,
        "Rollout_Size": args.rollout_size,
        "Minibatch_Size": args.minibatch_size,
        "PPO_Epochs": args.ppo_epochs,
        "Learning_Rate": args.learning_rate,
        "PPO_Clip": args.ppo_clip,
        "GAE_Lambda": args.gae_lambda,
        "Gamma": args.gamma,
        "Entropy_Coef": args.entropy_coef,
        "Exploration_Mode": "softmax_plus_uniform_mixture",
        "Exploration_Mix": args.exploration_mix,
        "Destroy_Min_Action_Probability": args.exploration_mix / ACTION_DIMS[0],
        "Repair_Min_Action_Probability": args.exploration_mix / ACTION_DIMS[1],
        "Severity_Min_Action_Probability": args.exploration_mix / ACTION_DIMS[2],
        "Temperature_Min_Action_Probability": args.exploration_mix / ACTION_DIMS[3],
        "Value_Coef": args.value_coef,
        "Max_Grad_Norm": args.max_grad_norm,
        "New_Best_Count": int(sum(reward_history) / NEW_BEST_REWARD),
        "Iterations": num_steps,
        "Device": str(DEVICE),
        "Seed": args.seed,
        "Destroy_Op1_Selections": int(action_counts[0][0]),
        "Destroy_Op2_Selections": int(action_counts[0][1]),
        "Destroy_Op3_Selections": int(action_counts[0][2]),
        "Destroy_Op4_Selections": int(action_counts[0][3]),
        "Destroy_Op5_Selections": int(action_counts[0][4]),
        "Repair_Op1_Selections": int(action_counts[1][0]),
        "Repair_Op2_Selections": int(action_counts[1][1]),
        "Repair_Op3_Selections": int(action_counts[1][2]),
        "Repair_Op4_Selections": int(action_counts[1][3]),
        "Repair_Op5_Selections": int(action_counts[1][4]),
    }
    save_to_csv(
        parent_path / "DRL_PPO_ALNS_Summary_Results.csv",
        summary_record,
    )

    log_dir = parent_path / "LOG"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"{file_name}_Reijnen_style_iteration_log_"
        f"{model.best_sol.obj:.2f}_{run_time:.2f}.json"
    )
    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(all_iterations, file, ensure_ascii=False, indent=2)

    result_summary_path = parent_path / f"{parent_path.name}_result.json"
    safe_append_to_json(
        result_summary_path,
        file_name,
        round(float(model.best_sol.obj), 2),
        round(run_time, 2),
    )

    print("\n--- Final Statistics ---")
    print(f"Best Objective Found: {model.best_sol.obj:.2f}")
    print(f"Total Run Time: {run_time:.2f}s")
    print(f"Online PPO Updates: {online_updates}")
    print(f"PPO Gradient Steps: {ppo_gradient_steps}")
    print(f"Learning Overhead: {learning_overhead:.2f}s ({overhead_ratio:.2f}%)")
    print("Destroy selections:", action_counts[0].tolist())
    print("Repair selections: ", action_counts[1].tolist())
    print("Severity selections:", action_counts[2].tolist())
    print("------------------------\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reijnen-style four-action PPO online ALNS for local CVRP"
    )
    parser.add_argument("--instance_path", required=True, help="CVRP CSV instance")
    parser.add_argument("--result_path", required=True, help="result file path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--rollout_size",
        type=int,
        default=8,
        help="ALNS iterations per online PPO update event",
    )
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument(
        "--exploration_mix",
        type=float,
        default=0.2,
        help="uniform exploration mass mixed into every categorical action head",
    )
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    args = parser.parse_args()

    if args.rollout_size <= 0 or args.minibatch_size <= 0:
        parser.error("rollout and minibatch sizes must be positive")
    if args.ppo_epochs <= 0:
        parser.error("--ppo_epochs must be positive")
    if not 0.0 <= args.exploration_mix < 1.0:
        parser.error("--exploration_mix must be in [0, 1)")
    if not 0.0 < args.ppo_clip < 1.0:
        parser.error("--ppo_clip must be in (0, 1)")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae_lambda must be in [0, 1]")
    return args


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    run(parse_args())
