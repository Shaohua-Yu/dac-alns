# -*- coding: utf-8 -*-
"""
DAC-ALNS for VRPTW (Duel-Actor + unified Critic)
------------------------------------------------
• 一体化可运行脚本：基于你提供的 VRPTW ALNS 评价函数/时间窗可行性与 Duel_AC(CVRP) 的智能自适应逻辑改写。
• 关键点：
  - 仍使用 VRPTW 的时间窗可行性拆分（JIT DP + 时刻表），保证修复后解可评估；
  - Duel-AC 两阶段（破坏/修复）各有 Actor，共享 GNN 编码器与 Critic；
  - 神经网络结点特征显式加入时间窗影响：ready/due/service、窗宽、从车场出发的“最早到达等待”“到期紧迫度”等；
  - 图包含静态基边(kNN 或全图) + 当前解路径动态边；
  - 破坏后的部分解作为“中间状态”，采取双阶段 TD 更新。

使用示例：
    python DAC_ALNS_VRPTW.py ./CAPP/pro_data/TW/C101.csv ./results/C101_run1.json --knn_k 8 --weight_mode inv --eps 0.1

依赖：numpy, torch, torch_geometric, matplotlib, numba(可选), xlsxwriter(仅用于输出xlsx，可省略)。
"""

import os, csv, math, time, json, copy, random, argparse
from pathlib import Path

import numpy as np

# ----- 可选：Numba 加速 -----
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

# ----- 可选：绘图后端（无显示环境兼容） -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- 深度学习组件 -----
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

torch.set_num_threads(1)

def set_seed(seed:int=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 基础数据结构（沿用 VRPTW 思路）
# ==============================
class Sol:
    def __init__(self):
        self.obj = None
        self.node_no_seq = []     # 仅客户 id 的序列
        self.route_list = []      # 含 depot 的路径列表 [[0, a, b, 0], ...]
        self.timetable_list = []  # 与上面一一对应
        self.route_distance = []  # 每条路线距离

class Node:
    def __init__(self):
        self.id = 0
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.demand = 0.0
        self.start_time = 0.0
        self.end_time = 1440.0
        self.service_time = 0.0

class Depot:
    def __init__(self):
        self.id = 0
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.start_time = 0.0
        self.end_time = 1440.0
        self.v_speed = 1.0
        self.v_cap = 80.0

class Model:
    def __init__(self):
        # 问题数据
        self.depot: Depot = None
        self.demand_dict = {}         # id->Node（含 depot.id）
        self.demand_id_list = []      # 仅客户 id（不含 depot）
        self.number_of_demands = 0

        # 距离/时间矩阵（dict/ndarray 双轨，便于复用）
        self.distance_matrix = {}
        self.time_matrix = {}
        # 数组版缓存
        self._id_list = None
        self._id_to_idx = None
        self._idx_to_id = None
        self._x = None; self._y = None
        self._demand_arr = None
        self._start_arr = None
        self._end_arr = None
        self._service_arr = None
        self._dist_mat_arr = None
        self._time_mat_arr = None

        # 进化参数 & 统计
        self.best_sol: Sol = None
        self.rand_d_max = 0.4; self.rand_d_min = 0.1
        self.worst_d_min = 5;   self.worst_d_max = 20
        self.regret_n = 2
        self.r1, self.r2, self.r3 = 20, 10, -0.1
        self.rho = 0.6
        self.T0 = 50.0; self.phi = 0.995
        # Duel-AC 奖励超参
        self.alpha1, self.alpha2 = 0.0, 1.0
        self.beta, self.delta = 1.0, 0.1
        # 计数
        self.d_select = np.zeros(2)
        self.r_select = np.zeros(3)

# ==============================
# 数据读取 & 矩阵
# ==============================

def _first_not_empty(row, keys, default=None):
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return row[k]
    return default


def read_single_csv(instance_file: str, model: Model):
    depot = Depot()
    with open(instance_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=1):
            if 'id' not in row or 'x_coord' not in row or 'y_coord' not in row:
                raise ValueError(f"CSV缺少必要列(id/x_coord/y_coord)，行{line_no}")
            rid = int(float(row['id']))
            is_depot = str(row.get('is_depot', '')).strip().lower() in ('1','true','yes','y')
            if is_depot or rid == 0:
                depot.id = rid
                depot.x_coord = float(row['x_coord'])
                depot.y_coord = float(row['y_coord'])
                depot.start_time = float(_first_not_empty(row, ['depot_start_time','ready_time','start_time'], 0))
                depot.end_time   = float(_first_not_empty(row, ['depot_end_time','due_time','end_time'], 1440))
                v_speed = _first_not_empty(row, ['v_speed'], None)
                v_cap   = _first_not_empty(row, ['v_cap'], None)
                depot.v_speed = float(v_speed) if v_speed is not None and float(v_speed) > 0 else 1.0
                depot.v_cap   = float(v_cap) if v_cap is not None else 80.0
                model.depot = depot
                model.demand_dict[depot.id] = depot
            else:
                node = Node()
                node.id = rid
                node.x_coord = float(row['x_coord'])
                node.y_coord = float(row['y_coord'])
                node.demand = float(_first_not_empty(row, ['demand'], 0))
                node.start_time = float(_first_not_empty(row, ['ready_time','start_time'], 0))
                node.end_time = float(_first_not_empty(row, ['due_time','end_time'], 1440))
                node.service_time = float(_first_not_empty(row, ['service_time'], 0))
                model.demand_dict[node.id] = node
                model.demand_id_list.append(node.id)
    if model.depot is None:
        raise ValueError("未在CSV中找到车场（id=0 或 is_depot=True 的行）。")
    if len(model.demand_id_list) == 0:
        raise ValueError("CSV中未发现任何客户行（id>=1）。")
    model.number_of_demands = len(model.demand_id_list)


def calDistanceTimeMatrix(model: Model):
    id_list = [model.depot.id] + list(model.demand_id_list)
    M = len(id_list)

    id_to_idx = {nid: i for i, nid in enumerate(id_list)}
    idx_to_id = np.array(id_list, dtype=np.int64)

    x = np.zeros(M); y = np.zeros(M)
    demand_arr = np.zeros(M)
    start_arr = np.zeros(M); end_arr = np.zeros(M)
    service_arr = np.zeros(M)

    x[0] = model.depot.x_coord; y[0] = model.depot.y_coord
    start_arr[0] = model.depot.start_time; end_arr[0] = model.depot.end_time
    demand_arr[0] = 0.0; service_arr[0] = 0.0

    for k, nid in enumerate(model.demand_id_list, start=1):
        n = model.demand_dict[nid]
        x[k] = n.x_coord; y[k] = n.y_coord
        demand_arr[k] = n.demand
        start_arr[k] = n.start_time; end_arr[k] = n.end_time
        service_arr[k] = n.service_time

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_mat = np.hypot(dx, dy)
    v = max(float(model.depot.v_speed), 1e-9)
    time_mat = dist_mat / v

    # 旧接口（dict）
    model.distance_matrix.clear(); model.time_matrix.clear()
    for i in range(M):
        idi = id_list[i]
        for j in range(M):
            idj = id_list[j]
            model.distance_matrix[(idi, idj)] = float(dist_mat[i, j])
            model.time_matrix[(idi, idj)] = float(time_mat[i, j])

    # 数组缓存
    model._id_list = id_list
    model._id_to_idx = id_to_idx
    model._idx_to_id = idx_to_id
    model._x = x; model._y = y
    model._demand_arr = demand_arr
    model._start_arr = start_arr; model._end_arr = end_arr
    model._service_arr = service_arr
    model._dist_mat_arr = dist_mat
    model._time_mat_arr = time_mat


def validate_single_customer_feasible(model: Model):
    infeasible = []
    for nid in model.demand_id_list:
        n = model.demand_dict[nid]
        t_go = model.time_matrix[(model.depot.id, nid)]
        t_back = model.time_matrix[(nid, model.depot.id)]
        arrival = max(model.depot.start_time + t_go, n.start_time)
        depart = arrival + n.service_time
        ok = (
            n.demand <= model.depot.v_cap and
            arrival <= n.end_time and
            depart + t_back <= model.depot.end_time
        )
        if not ok:
            infeasible.append(nid)
    if infeasible:
        raise ValueError("以下客户单独访问也不可行（时间窗/容量/速度设置不一致）：" + str(infeasible))

# ==============================
# VRPTW 时刻表/拆分（JIT）
# ==============================
@njit(cache=True, fastmath=True)
def _jit_split_routes_pred(seq_idx,
                           demand, start_t, end_t, service_t,
                           dist_mat, time_mat,
                           depot_idx, depot_start, depot_end, capacity):
    M = start_t.shape[0]
    INF = 1e18
    SENTINEL = -10**9  # 严格小于 -1（-1 可能是合法的“从车场出发”）

    V = np.empty(M, dtype=np.float64)
    for z in range(M):
        V[z] = INF
    V[depot_idx] = 0.0

    PredPos = np.empty(M, dtype=np.int64)
    for z in range(M):
        PredPos[z] = SENTINEL

    L = seq_idx.shape[0]
    for i in range(L):
        load = 0.0
        departure = 0.0
        cost = 0.0
        j = i
        while j < L:
            n2 = seq_idx[j]
            load += demand[n2]

            if j == i:
                # 车场 -> 首客户
                arr_tmp = depot_start + time_mat[depot_idx, n2]
                st = start_t[n2]
                arrival = arr_tmp if arr_tmp >= st else st
                # 首客户代价 = depot->n2 + n2->depot
                cost = dist_mat[depot_idx, n2] + dist_mat[n2, depot_idx]
            else:
                # 上一客户 -> 当前客户
                n3 = seq_idx[j - 1]
                arr_tmp = departure + time_mat[n3, n2]
                st = start_t[n2]
                arrival = arr_tmp if arr_tmp >= st else st
                # 用 prev->n2 + n2->depot 替换掉原来的 prev->depot
                cost = cost - dist_mat[n3, depot_idx] + dist_mat[n3, n2] + dist_mat[n2, depot_idx]

            departure = arrival + service_t[n2]

            feasible = (load <= capacity) and (arrival <= end_t[n2]) and \
                       (departure + time_mat[n2, depot_idx] <= depot_end)

            if feasible:
                n4 = seq_idx[i - 1] if i >= 1 else depot_idx
                new_val = V[n4] + cost
                if new_val <= V[n2]:
                    V[n2] = new_val
                    PredPos[n2] = i - 1
                j += 1
            else:
                # 关键修复：一旦不可行，停止扩展当前段
                break

    return PredPos


def extractRoutes(node_no_seq, P, depot_id):
    route_list = []
    j = len(node_no_seq)
    while j > 0:
        key = node_no_seq[j-1]
        i = P.get(key, j - 2)
        if i < -1: i = -1
        route = [depot_id]
        route.extend(node_no_seq[k] for k in range(i+1, j))
        route.append(depot_id)
        route_list.append(route)
        j = i + 1
    return route_list


def splitRoutes(node_no_seq, model: Model):
    id_to_idx = model._id_to_idx; depot = model.depot
    seq_idx = np.array([id_to_idx[nid] for nid in node_no_seq], dtype=np.int64)
    pred_pos_arr = _jit_split_routes_pred(
        seq_idx,
        model._demand_arr, model._start_arr, model._end_arr, model._service_arr,
        model._dist_mat_arr, model._time_mat_arr,
        id_to_idx[depot.id], depot.start_time, depot.end_time, float(depot.v_cap)
    )
    P = {}
    for pos, nid in enumerate(node_no_seq):
        idx = id_to_idx[nid]
        pred_pos = int(pred_pos_arr[idx])
        if pred_pos >= -1: P[nid] = pred_pos
    return extractRoutes(node_no_seq, P, depot.id)


def calTravelCost(route_list, model: Model):
    timetable_list = []
    route_distance = []
    total_distance = 0.0
    for route in route_list:
        timetable = []
        distance = 0.0
        for i in range(len(route)):
            if i == 0:
                depot_id = route[i]; nxt = route[i+1]
                travel_time = model.time_matrix[(depot_id, nxt)]
                departure = max(model.depot.start_time, model.demand_dict[nxt].start_time - travel_time)
                timetable.append((departure, departure))
            elif 1 <= i <= len(route)-2:
                last = route[i-1]; cur = route[i]
                travel_time = model.time_matrix[(last, cur)]
                arrival = max(timetable[-1][1] + travel_time, model.demand_dict[cur].start_time)
                departure = arrival + model.demand_dict[cur].service_time
                timetable.append((arrival, departure))
                distance += model.distance_matrix[(last, cur)]
            else:
                last = route[i-1]; depot_id = route[i]
                travel_time = model.time_matrix[(last, depot_id)]
                departure = timetable[-1][1] + travel_time
                timetable.append((departure, departure))
                distance += model.distance_matrix[(last, depot_id)]
        total_distance += distance
        route_distance.append(distance)
        timetable_list.append(timetable)
    return timetable_list, total_distance, route_distance


def calObj(node_no_seq, model: Model):
    seq = copy.deepcopy(node_no_seq)
    route_list = splitRoutes(seq, model)
    timetables, cost, route_distance = calTravelCost(route_list, model)
    return cost, route_list, route_distance, timetables

# ==============================
# 破坏 / 修复算子（沿用 VRPTW 位置索引 remove_list 约定）
# ==============================

def genInitialSol(node_no_seq):
    arr = copy.deepcopy(node_no_seq); random.shuffle(arr); return arr

def createRandomDestory(model: Model, sol: 'Sol' = None):  # sol未用，仅为统一签名
    d = random.uniform(model.rand_d_min, model.rand_d_max)
    return random.sample(range(model.number_of_demands), max(1, int(d*(model.number_of_demands-1))))

def createWorseDestory(model: Model, sol: Sol):
    deta_f = []
    for node_no in sol.node_no_seq:
        tmp = copy.deepcopy(sol.node_no_seq)
        tmp.remove(node_no)
        obj,_,_,_ = calObj(tmp, model)
        deta_f.append(sol.obj - obj)
    sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
    d = max(1, min(model.worst_d_max, max(model.worst_d_min, len(sorted_id))))
    return sorted_id[:d]

def createRandomRepair(remove_list, model: Model, sol: Sol):
    unassigned = []
    assigned = []
    for i in range(model.number_of_demands):
        (unassigned if i in remove_list else assigned).append(sol.node_no_seq[i])
    for node_no in unassigned:
        idx = random.randint(0, max(0, len(assigned)-1))
        assigned.insert(idx, node_no)
    new_sol = Sol(); new_sol.node_no_seq = copy.deepcopy(assigned)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list = calObj(assigned, model)
    return new_sol

def _obj_of(seq, model):
    return calObj(seq, model)[0]

def findGreedyInsert(unassigned, assigned, model: Model):
    best_node, best_idx, best_cost = None, 0, float('inf')
    base_obj = _obj_of(assigned, model)
    for node_no in unassigned:
        for i in range(len(assigned)+1):
            tmp = assigned[:]; tmp.insert(i, node_no)
            obj = _obj_of(tmp, model); delta = obj - base_obj
            if delta < best_cost:
                best_cost = delta; best_idx = i; best_node = node_no
    return best_node, best_idx

def createGreedyRepair(remove_list, model: Model, sol: Sol):
    unassigned = []
    assigned = []
    for i in range(model.number_of_demands):
        (unassigned if i in remove_list else assigned).append(sol.node_no_seq[i])
    while unassigned:
        node, idx = findGreedyInsert(unassigned, assigned, model)
        assigned.insert(idx, node); unassigned.remove(node)
    new_sol = Sol(); new_sol.node_no_seq = copy.deepcopy(assigned)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list = calObj(assigned, model)
    return new_sol

def findRegretInsert(unassigned, assigned, model: Model, k_regret:int=2):
    opt_node, opt_idx, opt_regret = None, 0, -float('inf')
    base_obj = _obj_of(assigned, model)
    for node_no in unassigned:
        cand = []
        for i in range(len(assigned)+1):
            tmp = assigned[:]; tmp.insert(i, node_no)
            obj = _obj_of(tmp, model); cand.append((obj-base_obj, i))
        cand.sort(key=lambda x:x[0])
        if cand:
            regret = 0.0
            for t in range(1, min(k_regret, len(cand))):
                regret += cand[t][0] - cand[0][0]
            if regret > opt_regret:
                opt_regret = regret; opt_node = node_no; opt_idx = cand[0][1]
    return opt_node, opt_idx

def createRegretRepair(remove_list, model: Model, sol: Sol):
    unassigned = []
    assigned = []
    for i in range(model.number_of_demands):
        (unassigned if i in remove_list else assigned).append(sol.node_no_seq[i])
    while unassigned:
        node, idx = findRegretInsert(unassigned, assigned, model, k_regret=model.regret_n)
        if node is None:
            node, idx = findGreedyInsert(unassigned, assigned, model)
        assigned.insert(idx, node); unassigned.remove(node)
    new_sol = Sol(); new_sol.node_no_seq = copy.deepcopy(assigned)
    new_sol.obj, new_sol.route_list, new_sol.route_distance, new_sol.timetable_list = calObj(assigned, model)
    return new_sol

# 包装：按 id 选择算子
DESTROY_FUNCS = (createRandomDestory, createWorseDestory)
REPAIR_FUNCS  = (createRandomRepair, createGreedyRepair, createRegretRepair)

def doDestroy_by_id(did, model: Model, sol: Sol):
    return DESTROY_FUNCS[did](model, sol)

def doRepair_by_id(rid, remove_list, model: Model, sol: Sol):
    return REPAIR_FUNCS[rid](remove_list, model, sol)

def apply_destroy_to_partial_sol(sol: Sol, remove_list, model: Model):
    part = Sol(); keep = [x for i,x in enumerate(sol.node_no_seq) if i not in set(remove_list)]
    part.node_no_seq = keep
    part.obj, part.route_list, part.route_distance, part.timetable_list = calObj(keep, model)
    return part

# ==============================
# 图构建（含时间窗特征）
# ==============================

def precompute_base_graph(model: Model, k_nn:int=10, use_full_graph_knn:bool=False, weight_mode:str='inv'):
    n = len(model._id_list)  # 含 depot
    edges = set()
    if use_full_graph_knn:
        for u in range(n):
            for v in range(n):
                if u != v: edges.add((u, v))
    else:
        k = min(k_nn, max(1, n-1))
        for u in range(n):
            dists = [(v, model._dist_mat_arr[u, v]) for v in range(n) if v != u]
            dists.sort(key=lambda x: x[1])
            for v, _ in dists[:k]:
                edges.add((u, v)); edges.add((v, u))
    if not edges:
        ei = torch.empty((2,0), dtype=torch.long, device=device)
        ew = torch.empty((0,), dtype=torch.float, device=device)
        return ei, ew
    ei = torch.tensor(list(edges), dtype=torch.long, device=device).t().contiguous()
    if weight_mode == 'dist':
        ew = torch.tensor([float(model._dist_mat_arr[u, v]) for (u, v) in edges], dtype=torch.float, device=device)
    else:
        ew = torch.tensor([1.0/(float(model._dist_mat_arr[u, v]) + 1e-6) for (u, v) in edges], dtype=torch.float, device=device)
    return ei, ew


def _node_features_with_tw(model: Model, removed_idx_set=None):
    # 结点特征：x,y,demand,served_flag, start,end,width,service, tightness, wait
    n = len(model._id_list)
    served = np.ones(n, dtype=np.float32)
    if removed_idx_set:
        for idx in removed_idx_set:
            if 0 <= idx < n: served[idx] = 0.0
    dep_idx = model._id_to_idx[model.depot.id]
    horizon = max(1e-6, model.depot.end_time - model.depot.start_time)
    earliest = model.depot.start_time + model._time_mat_arr[dep_idx]  # 到各点“最早出发到达”

    start = model._start_arr; end = model._end_arr; service = model._service_arr
    width = end - start
    wait = np.maximum(0.0, start - earliest)
    tight = end - earliest

    # 规范化：空间坐标/需求不做缩放（可在数据规模前处理），时间相关按 horizon 归一
    feats = np.stack([
        model._x.astype(np.float32),
        model._y.astype(np.float32),
        model._demand_arr.astype(np.float32),
        served.astype(np.float32),
        (start - model.depot.start_time).astype(np.float32)/horizon,
        (end   - model.depot.start_time).astype(np.float32)/horizon,
        (width / max(1e-6, width.max())).astype(np.float32),
        (service / max(1e-6, service.max())).astype(np.float32),
        (tight / horizon).astype(np.float32),
        (wait  / horizon).astype(np.float32),
    ], axis=1)
    return torch.tensor(feats, dtype=torch.float, device=device)


def build_graph(sol: Sol, model: Model, base_edge_index, base_edge_weight, removed_idx_set=None, include_path_edges=True):
    x = _node_features_with_tw(model, removed_idx_set=removed_idx_set)
    edge_index = base_edge_index; edge_weight = base_edge_weight
    if include_path_edges and sol.route_list:
        dyn_edges = set()
        for route in sol.route_list:
            for i in range(len(route)-1):
                u_id, v_id = route[i], route[i+1]
                u = model._id_to_idx[u_id]; v = model._id_to_idx[v_id]
                dyn_edges.add((u, v)); dyn_edges.add((v, u))
        if dyn_edges:
            ei2 = torch.tensor(list(dyn_edges), dtype=torch.long, device=device).t().contiguous()
            ew2 = torch.tensor([1.0/(model._dist_mat_arr[u, v] + 1e-6) for (u, v) in dyn_edges], dtype=torch.float, device=device)
            edge_index = torch.cat([edge_index, ei2], dim=1)
            edge_weight = torch.cat([edge_weight, ew2], dim=0)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

# ==============================
# GNN 编码器 & Duel-AC Agent（输入维按 TW 扩展）
# ==============================
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=10, hid=64, out_dim=64, use_attention=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.out_dim = out_dim
        self.use_attention = use_attention
        self.score = torch.nn.Linear(out_dim, 1) if use_attention else None
        self.fallback = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim)
        )
    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        w = getattr(data, 'edge_weight', None)
        if edge_index is None or edge_index.numel() == 0:
            h = self.fallback(x)
        else:
            h = F.relu(self.conv1(x, edge_index, w))
            h = self.conv2(h, edge_index, w)
        if self.use_attention:
            scores = self.score(h).squeeze(-1)
            alpha = torch.softmax(scores, dim=0)
            g_avg = (alpha.unsqueeze(-1) * h).sum(dim=0, keepdim=True)
        else:
            g_avg = h.mean(dim=0, keepdim=True)
        g_max, _ = h.max(dim=0, keepdim=True)
        return torch.cat([g_avg, g_max], dim=-1)  # [1, 2*out_dim]

class DACAgent(torch.nn.Module):
    def __init__(self, in_dim=10, enc_dim=64, extra_state_dim=2, hid=64, gamma=0.9, ent_coef=0.01):
        super().__init__()
        self.encoder = GraphEncoder(in_dim=in_dim, hid=hid, out_dim=enc_dim, use_attention=True).to(device)
        enc_feat_dim = self.encoder.out_dim * 2
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1)
        ).to(device)
        self.actor_destroy = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 2)
        ).to(device)
        self.actor_repair = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 3)
        ).to(device)
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.opt_ad = torch.optim.Adam(self.actor_destroy.parameters(), lr=1e-3)
        self.opt_ar = torch.optim.Adam(self.actor_repair.parameters(), lr=1e-3)
        self.gamma = gamma; self.ent_coef = ent_coef; self.n_destroy = 2
    def build_state(self, graph: Data, ratio: float, destroy_id:int=0):
        g = self.encoder(graph)
        r = torch.tensor([[float(ratio)]], device=device, dtype=g.dtype)
        d_norm = float(destroy_id)/(self.n_destroy-1) if self.n_destroy>1 else 0.0
        d = torch.tensor([[d_norm]], device=device, dtype=g.dtype)
        return torch.cat([g, r, d], dim=-1)
    def _sample_with_eps(self, logits, eps, n_actions):
        if random.random() < eps: return random.randrange(n_actions)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()
    def act_destroy(self, state, eps=0.1):
        return self._sample_with_eps(self.actor_destroy(state), eps, 2)
    def act_repair(self, state, eps=0.1):
        return self._sample_with_eps(self.actor_repair(state), eps, 3)
    def update(self, d_trans, r_trans):
        sd  = torch.cat(d_trans['states'], dim=0).to(device)
        ad  = torch.tensor(d_trans['actions'], dtype=torch.long, device=device).view(-1,1)
        rd  = torch.tensor(d_trans['rewards'], dtype=torch.float, device=device).view(-1,1)
        sdn = torch.cat(d_trans['next_states'], dim=0).to(device)
        dd  = torch.tensor(d_trans['dones'], dtype=torch.float, device=device).view(-1,1)
        sr  = torch.cat(r_trans['states'], dim=0).to(device)
        ar  = torch.tensor(r_trans['actions'], dtype=torch.long, device=device).view(-1,1)
        rr  = torch.tensor(r_trans['rewards'], dtype=torch.float, device=device).view(-1,1)
        srn = torch.cat(r_trans['next_states'], dim=0).to(device)
        dr  = torch.tensor(r_trans['dones'], dtype=torch.float, device=device).view(-1,1)
        with torch.no_grad():
            vdn = self.critic(sdn); vrn = self.critic(srn)
            td_tgt_d = rd + self.gamma * vdn * (1 - dd)
            td_tgt_r = rr + self.gamma * vrn * (1 - dr)
        v_d = self.critic(sd); v_r = self.critic(sr)
        critic_loss = F.mse_loss(v_d, td_tgt_d) + F.mse_loss(v_r, td_tgt_r)
        td_delta_d = (td_tgt_d - v_d).detach(); td_delta_r = (td_tgt_r - v_r).detach()
        logits_d = self.actor_destroy(sd); logits_r = self.actor_repair(sr)
        logp_d = torch.log(F.softmax(logits_d, dim=1).gather(1, ad) + 1e-8)
        logp_r = torch.log(F.softmax(logits_r, dim=1).gather(1, ar) + 1e-8)
        ent_d = -(F.softmax(logits_d, dim=1) * F.log_softmax(logits_d, dim=1)).sum(dim=1, keepdim=True).mean()
        ent_r = -(F.softmax(logits_r, dim=1) * F.log_softmax(logits_r, dim=1)).sum(dim=1, keepdim=True).mean()
        actor_d_loss = torch.mean(-logp_d * td_delta_d) - self.ent_coef * ent_d
        actor_r_loss = torch.mean(-logp_r * td_delta_r) - self.ent_coef * ent_r
        total = critic_loss + actor_d_loss + actor_r_loss
        self.opt_enc.zero_grad(); self.opt_critic.zero_grad(); self.opt_ad.zero_grad(); self.opt_ar.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.critic.parameters()) + list(self.actor_destroy.parameters()) + list(self.actor_repair.parameters()), 1.0)
        self.opt_enc.step(); self.opt_critic.step(); self.opt_ad.step(); self.opt_ar.step()

# ==============================
# 可视化/输出
# ==============================

def plotObj(obj_list):
    try:
        plt.figure(); plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
        plt.xlabel('Iterations'); plt.ylabel('Best Obj'); plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(); plt.show()
    except Exception:
        pass

# ==============================
# 主程序（Duel-Stage TD）
# ==============================
parser = argparse.ArgumentParser(description='DAC-ALNS for VRPTW (time windows aware)')
parser.add_argument('file_path', type=str, help='CSV文件路径（含车场与所有客户）')
parser.add_argument('result_file_path', type=str, help='结果聚合JSON目标路径（用于所在目录聚合/存模）')
parser.add_argument('--knn_k', type=int, default=8, help='kNN 邻居数（静态基边）')
parser.add_argument('--full_graph_knn', action='store_true', help='使用全连接图替代 kNN')
parser.add_argument('--weight_mode', choices=['inv','dist'], default='inv', help='边权：inv(1/d) 或 dist(d)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=0.10, help='epsilon-greedy 探索概率')
parser.add_argument('--epochs', type=int, default=0, help='已废弃参数，仅为兼容')
args = parser.parse_args()

set_seed(args.seed)

# 超参（可按实例规模调整）
rand_d_max, rand_d_min = 0.4, 0.1
worst_d_min, worst_d_max = 5, 20
regret_n = 2
r1, r2, r3 = 20, 10, -0.1
rho = 0.6
# T0 = 50.0
phi = 0.995

gamma = 0.9
ent_coef = 0.01

# 初始化模型
model = Model()
model.rand_d_max, model.rand_d_min = rand_d_max, rand_d_min
model.worst_d_min, model.worst_d_max = worst_d_min, worst_d_max
model.regret_n = regret_n
model.r1, model.r2, model.r3 = r1, r2, r3
model.rho = rho
model.phi = phi

read_single_csv(args.file_path, model)
calDistanceTimeMatrix(model)
validate_single_customer_feasible(model)

# 初解
sol = Sol()
sol.node_no_seq = genInitialSol(model.demand_id_list)
sol.obj, sol.route_list, sol.route_distance, sol.timetable_list = calObj(sol.node_no_seq, model)
model.best_sol = copy.deepcopy(sol)
model.T0 = sol.obj * 0.2
# 图基边
base_edge_index, base_edge_weight = precompute_base_graph(model, k_nn=args.knn_k, use_full_graph_knn=args.full_graph_knn, weight_mode=args.weight_mode)

# Agent（结点特征维度 = 10）
agent = DACAgent(in_dim=10, enc_dim=64, extra_state_dim=2, hid=64, gamma=gamma, ent_coef=ent_coef).to(device)

history_best_obj = [sol.obj]
start_time = time.time()
# num_steps = max(50 * model.number_of_demands, 2000)
num_steps = max(50 * model.number_of_demands, 2000)
all_iterations = []

for it in range(num_steps):
    # 完整解状态
    g_s = build_graph(sol, model, base_edge_index, base_edge_weight, removed_idx_set=None, include_path_edges=True)
    s = agent.build_state(g_s, ratio=0.0, destroy_id=0)

    # 破坏
    a_d = agent.act_destroy(s, eps=args.eps); model.d_select[a_d] += 1
    remove_list = doDestroy_by_id(a_d, model, sol)  # 位置索引列表
    des_ratio = len(remove_list) / max(1, model.number_of_demands)

    # 破坏后部分解
    part_sol = apply_destroy_to_partial_sol(sol, remove_list, model)
    removed_idx_set = set(model._id_to_idx[nid] for nid in sol.node_no_seq if nid not in part_sol.node_no_seq)
    g_sd = build_graph(part_sol, model, base_edge_index, base_edge_weight, removed_idx_set=removed_idx_set, include_path_edges=True)
    s_d = agent.build_state(g_sd, ratio=des_ratio, destroy_id=a_d)

    # 修复
    a_r = agent.act_repair(s_d, eps=args.eps); model.r_select[a_r] += 1
    new_sol = doRepair_by_id(a_r, remove_list, model, sol)

    # 新完整解状态
    g_sp = build_graph(new_sol, model, base_edge_index, base_edge_weight, removed_idx_set=None, include_path_edges=True)
    s_prime = agent.build_state(g_sp, ratio=des_ratio, destroy_id=a_d)

    # ----- 奖励（双阶段） -----
    C_best, C_before, C_destroy, C_repair = model.best_sol.obj, sol.obj, part_sol.obj, new_sol.obj
    impr_ratio_d = (C_best - C_destroy) / max(1e-6, C_best)
    if C_best < C_destroy < C_before:
        d_reward = model.alpha1 + model.beta * impr_ratio_d
    elif C_destroy <= C_best:
        d_reward = model.alpha2 + model.beta * impr_ratio_d
    else:
        d_reward = 0.0

    f_rep_id = model.r_select[a_r]; f_max = max(1.0, model.r_select.max())
    explore_bonus = model.delta * (1.0 - f_rep_id / f_max)
    if C_repair < C_best:
        r_reward = model.r1 + explore_bonus
    elif C_best <= C_repair < C_before:
        r_reward = model.r2 + explore_bonus
    else:
        r_reward = model.r3 + explore_bonus

    # ----- 接纳（退火） -----
    accepted = False
    if C_repair < C_before:
        sol = copy.deepcopy(new_sol); accepted = True
        if C_repair < model.best_sol.obj:
            model.best_sol = copy.deepcopy(new_sol)
    elif math.exp((C_before - C_repair) / max(model.T0, 1e-3)) >= random.random():
        sol = copy.deepcopy(new_sol); accepted = True

    g_next = g_sp if accepted else g_s
    s_next = agent.build_state(g_next, ratio=des_ratio, destroy_id=a_d)

    d_trans = {'states':[s], 'actions':[a_d], 'rewards':[d_reward], 'next_states':[s_d], 'dones':[0.0]}
    r_trans = {'states':[s_d], 'actions':[a_r], 'rewards':[r_reward], 'next_states':[s_next], 'dones':[1.0]}

    model.T0 *= model.phi
    agent.update(d_trans, r_trans)

    # 日志
    run_time = time.time() - start_time
    history_best_obj.append(model.best_sol.obj)
    # if (it+1) % 100 == 0:
    if (it + 1) % 1 == 0:
        print(f"{Path(args.file_path).name} | Iter {it+1}/{num_steps} | Best Obj: {model.best_sol.obj:.2f} | Time: {run_time:.2f}s")
    if (it+1) % 10 == 0:
        all_iterations.append({
            'iteration': it+1,
            'file_path': args.file_path,
            'best_obj': round(model.best_sol.obj, 2),
            'run_time': round(run_time, 2),
            'rand_des': float(model.d_select[0]),
            'wors_des': float(model.d_select[1]),
            'rand_rep': float(model.r_select[0]),
            'gree_rep': float(model.r_select[1]),
            'regr_rep': float(model.r_select[2]),
        })

# 收尾：聚合/存模
parent_path = Path(args.result_file_path).parent
parts = Path(args.result_file_path).parts
file_name = Path(args.result_file_path).stem
log_dir = parent_path / 'LOG'; log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / f'{file_name}_iteration_log_{round(model.best_sol.obj, 2)}_{round(run_time, 2)}.json'
with open(log_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_iterations, f, indent=4, ensure_ascii=False)

try:
    result_path = parent_path / f'{parts[-2]}_result.json'
    data = {}
    if result_path.exists():
        try:
            with open(result_path, 'r', encoding='utf-8') as fr: data = json.load(fr)
        except json.JSONDecodeError: pass
    problem = data.setdefault(file_name, {"objectives": [], "times": []})
    problem["objectives"].append(round(model.best_sol.obj, 2)); problem["times"].append(round(run_time, 2))
    with open(result_path, 'w', encoding='utf-8') as fw: json.dump(data, fw, ensure_ascii=False, indent=2)
except IndexError:
    pass

try:
    model_dir = parent_path.parent.parent / 'SavedModel' / parts[-2]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f'{file_name}_model_{round(model.best_sol.obj, 2)}.pth'
    torch.save(agent.state_dict(), model_save_path)
except Exception:
    model_save_path = Path(f'./{file_name}_model_{round(model.best_sol.obj, 2)}.pth')
    torch.save(agent.state_dict(), model_save_path)

print("\n--- Final Statistics ---")
print(f"Best Objective Found: {model.best_sol.obj:.2f}")
print(f"Total Run Time: {run_time:.2f}s")
print(f"Random Destroy Selections: {model.d_select[0]}")
print(f"Worse Destroy Selections:  {model.d_select[1]}")
print(f"Random Repair Selections:  {model.r_select[0]}")
print(f"Greedy Repair Selections:  {model.r_select[1]}")
print(f"Regret  Repair Selections: {model.r_select[2]}")
print("------------------------\n")
