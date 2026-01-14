# -*- coding: utf-8 -*-
# DAC-ALNS (修正版)：路径边 + kNN/全图(预计算) + 距离权重
# 共享GNN编码器 + 破坏/修复双Actor + 统一Critic + 双阶段TD
# 关键修复：去除 no_grad 让编码器可训练；加入熵正则与epsilon探索；CPU下用较小hidden加速
# 用法：python DAC_ALNS.py <csv_path> <result_file_path> [--knn_k 10] [--full_graph_knn] [--weight_mode inv|dist] [--seed 0]

import os, csv, math, time, json, copy, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import matplotlib
matplotlib.use("Agg")  # 兼容无显示环境
import matplotlib.pyplot as plt
torch.set_num_threads(1)   # 限制 PyTorch 在 CPU 上使用的线程数

# ---------------------- 设备/随机种子 ---------------------- #
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- JSON 结果聚合 ---------------------- #
def safe_append_to_json(file_path, problem_name, obj, timestamp):
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass
    problem = data.setdefault(problem_name, {"objectives": [], "times": []})
    problem["objectives"].append(obj); problem["times"].append(timestamp)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------- 基础数据结构 ---------------------- #
class Sol:
    def __init__(self):
        self.node_no_seq = None
        self.obj = None
        self.route_list = None
        self.route_distance = None

class Node:
    def __init__(self):
        self.id = 0; self.x_coord = 0.0; self.y_coord = 0.0; self.demand = 0.0

class Model:
    def __init__(self):
        self.best_sol = None
        self.node_id_list = []     # 不含depot
        self.demand_dict = {}      # id->Node（含depot）
        self.depot = None
        self.number_of_demands = 0
        self.vehicle_cap = 100
        self.distance_matrix = {}
        self.distance_matrix_dict = {}

        # 算子与奖励
        self.rand_d_max = 0.4; self.rand_d_min = 0.1
        self.worst_d_min = 5;   self.worst_d_max = 20
        self.regret_n = 2
        self.r1, self.r2, self.r3 = 20, 10, -0.1
        self.rho = 0.6
        self.alpha1, self.alpha2 = 0.0, 1.0
        self.beta, self.delta = 1.0, 0.1
        # self.gamma1, self.gamma2, self.gamma3 = 1.0, 0.5, 0.0
        self.T0 = 50.0  # 初始温度，可按实例规模调
        self.phi = 0.995  # 冷却率（0.99~0.999）

        # 统计
        self.d_weight = np.ones(2) * 10
        self.d_select = np.zeros(2); self.d_score = np.zeros(2)
        self.d_history_select = np.zeros(2); self.d_history_score = np.zeros(2)
        self.r_weight = np.ones(3) * 10
        self.r_select = np.zeros(3); self.r_score = np.zeros(3)
        self.r_history_select = np.zeros(3); self.r_history_score = np.zeros(3)

# ---------------------- 读取/距离 ---------------------- #
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
                node.id = 0; model.depot = node
            else:
                node.id = int(float(row['id'])) - 1
                model.node_id_list.append(node.id)
            model.demand_dict[node.id] = node
    model.number_of_demands = len(model.demand_dict) - 1

def calDistanceMatrix(model):
    ids = [model.depot.id] + model.node_id_list              # [0, 1, 2, …]
    coords = np.array([[model.demand_dict[i].x_coord,
                        model.demand_dict[i].y_coord] for i in ids],
                      dtype=np.float32)                      # shape=(n+1,2)

    diff = coords[:, None, :] - coords[None, :, :]           # (n+1,n+1,2)
    dist_mat = np.sqrt((diff ** 2).sum(-1, dtype=np.float32))  # (n+1,n+1)

    model.distance_matrix = dist_mat                         # ndarray
    model.id2idx = {id_: idx for idx, id_ in enumerate(ids)}
    model.idx2id = ids                                        # list

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            model.distance_matrix_dict[(id_i, id_j)] = dist_mat[i, j] \
                if hasattr(model, "distance_matrix_dict") else None

def genInitialSol(node_no_seq):
    node_no_seq = copy.deepcopy(node_no_seq)
    random.shuffle(node_no_seq)
    return node_no_seq


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

    F    = np.full(n + 1, np.inf, dtype=np.float32)
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
                cur_idx  = seq[j - 1]
                cost = cost \
                       - dist_mat[prev_idx, depot_idx] \
                       + dist_mat[prev_idx, cur_idx]  \
                       + dist_mat[cur_idx,  depot_idx]

            new_cost = F[j - 1] + cost
            if new_cost < F[i]:
                F[i]   = new_cost
                pred[i]= j - 1
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
                       dtype=np.int32)

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

def createRandomDestory(model):
    if not model.node_id_list: return []
    num_nodes = len(model.node_id_list)
    if num_nodes <= 1: return []
    d = random.uniform(model.rand_d_min, model.rand_d_max)
    num_to_remove = min(max(1, int(d * num_nodes)), num_nodes - 1 if num_nodes > 1 else 0)
    if num_to_remove == 0 and num_nodes > 0: num_to_remove = 1
    if num_to_remove == 0 : return []
    k = min(num_to_remove, len(model.node_id_list))
    return random.sample(model.node_id_list, k)


def createWorseDestory(model, sol):
    deta_f = []
    if not sol.node_no_seq: return []
    current_obj = sol.obj
    if current_obj == float('inf'): return []
    nodes_to_consider = copy.deepcopy(sol.node_no_seq)
    for node_no in nodes_to_consider:
        node_no_seq_ = [n for n in nodes_to_consider if n != node_no]
        if not node_no_seq_:
             obj = 0
        else:
             obj, _, _ = calObj(node_no_seq_, model)
        if obj == float('inf'):
            delta = -float('inf')
        else:
            delta = obj - current_obj
            if obj == float('inf'):
                 delta = -float('inf')
            else:
                 delta = current_obj - obj
        deta_f.append(delta)
    node_delta_pairs = list(zip(nodes_to_consider, deta_f))
    node_delta_pairs.sort(key=lambda item: item[1], reverse=True)

    # Determine number to remove
    num_nodes = len(nodes_to_consider)
    d_min = min(model.worst_d_min, num_nodes - 1 if num_nodes > 1 else 0)
    d_max = min(model.worst_d_max, num_nodes - 1 if num_nodes > 1 else 0)
    if d_min < 0: d_min = 0
    if d_max < d_min: d_max = d_min
    if num_nodes > 1 and d_max == 0: d_max = 1 # Ensure at least 1 if possible
    if d_min > d_max : d_min = d_max

    num_to_remove = 0
    if d_max > 0:
        num_to_remove = random.randint(d_min, d_max)
        num_to_remove = min(num_to_remove, len(node_delta_pairs), num_nodes - 1 if num_nodes > 1 else 0)
        if num_to_remove < 0: num_to_remove = 0

    if num_to_remove == 0: return []

    # Select the top 'num_to_remove' nodes based on sorted delta_f
    return [pair[0] for pair in node_delta_pairs[:num_to_remove]]



def createRandomRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    for node_no in unassigned_node_no_seq:
        insert_limit = len(assigned_node_no_seq)
        index = random.randint(0, insert_limit)
        assigned_node_no_seq.insert(index, node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol


def findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    best_insert_node_no = None
    best_insert_index = None
    best_insert_cost_increase = float('inf')
    if not assigned_node_no_seq:
         base_obj = 0
    else:
        base_obj_result = calObj(assigned_node_no_seq, model)
        base_obj = base_obj_result[0]
        if base_obj == float('inf'):
            pass

    for node_no in unassigned_node_no_seq:
        node_best_index = -1
        node_best_cost_increase = float('inf')
        for i in range(len(assigned_node_no_seq) + 1):
            assigned_node_no_seq_ = assigned_node_no_seq[:i] + [node_no] + assigned_node_no_seq[i:]
            obj_, _, _ = calObj(assigned_node_no_seq_, model)
            if obj_ == float('inf'):
                cost_increase = float('inf')
            elif base_obj == float('inf'):
                cost_increase = -float('inf')
            else:
                cost_increase = obj_ - base_obj
            if cost_increase < node_best_cost_increase:
                node_best_cost_increase = cost_increase
                node_best_index = i
        if node_best_cost_increase < best_insert_cost_increase:
            best_insert_cost_increase = node_best_cost_increase
            best_insert_index = node_best_index
            best_insert_node_no = node_no

    if best_insert_node_no is None and unassigned_node_no_seq:
        return None, None

    return best_insert_node_no, best_insert_index



def createGreedyRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        if insert_node_no is None:
            break
        assigned_node_no_seq.insert(insert_index, insert_node_no)
        unassigned_node_no_seq.remove(insert_node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol


def findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model):
    opt_insert_node_no = None
    opt_insert_index = None
    opt_node_max_regret = -float('inf')

    if not assigned_node_no_seq:
         base_obj = 0
    else:
        base_obj_result = calObj(assigned_node_no_seq, model)
        base_obj = base_obj_result[0]
        if base_obj == float('inf'):
             pass
    candidate_insertions = {}

    for node_no in unassigned_node_no_seq:
        insertion_options = []
        for i in range(len(assigned_node_no_seq) + 1):
            assigned_node_no_seq_ = assigned_node_no_seq[:i] + [node_no] + assigned_node_no_seq[i:]
            obj_, _, _ = calObj(assigned_node_no_seq_, model)

            if obj_ == float('inf'):
                cost_increase = float('inf')
            elif base_obj == float('inf'):
                 cost_increase = -float('inf') if obj_ != float('inf') else float('inf')
            else:
                cost_increase = obj_ - base_obj

            if cost_increase != float('inf'):
                 insertion_options.append((cost_increase, i))
        insertion_options.sort(key=lambda x: x[0])

        if insertion_options:
             candidate_insertions[node_no] = insertion_options

    node_regrets = {}
    for node_no, options in candidate_insertions.items():
        regret_val = 0
        best_cost_increase = options[0][0]

        for k in range(1, min(model.regret_n, len(options))):
            regret_val += (options[k][0] - best_cost_increase)

        node_regrets[node_no] = regret_val

        if regret_val > opt_node_max_regret:
            opt_node_max_regret = regret_val
            opt_insert_node_no = node_no
            opt_insert_index = options[0][1]

    if opt_insert_node_no is None and unassigned_node_no_seq:
        min_cost_increase = float('inf')
        fallback_node = None
        fallback_index = None
        for node_no, options in candidate_insertions.items():
            if options and options[0][0] < min_cost_increase:
                min_cost_increase = options[0][0]
                fallback_node = node_no
                fallback_index = options[0][1]

        if fallback_node is not None:
             opt_insert_node_no = fallback_node
             opt_insert_index = fallback_index
        else:
             return None, None

    return opt_insert_node_no, opt_insert_index


def createRegretRepair(remove_list, model, sol):
    unassigned_node_no_seq = copy.deepcopy(remove_list)
    assigned_node_no_seq = [node_no for node_no in sol.node_no_seq if node_no not in unassigned_node_no_seq]

    while len(unassigned_node_no_seq) > 0:
        insert_node_no, insert_index = findRegretInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
        if insert_node_no is None:
            insert_node_no, insert_index = findGreedyInsert(unassigned_node_no_seq, assigned_node_no_seq, model)
            if insert_node_no is None:
                 break

        assigned_node_no_seq.insert(insert_index, insert_node_no)
        unassigned_node_no_seq.remove(insert_node_no)

    new_sol = Sol()
    new_sol.node_no_seq = assigned_node_no_seq
    new_sol.obj, new_sol.route_list, new_sol.route_distance = calObj(assigned_node_no_seq, model)
    return new_sol

def doDestroy_by_id(did, model: Model, sol: Sol):
    return createRandomDestory(model) if did == 0 else createWorseDestory(model, sol)

def doRepair_by_id(rid, remove_list, model: Model, sol: Sol):
    return (createRandomRepair, createGreedyRepair, createRegretRepair)[rid](remove_list, model, sol)

def apply_destroy_to_partial_sol(sol: Sol, remove_list, model: Model):
    part = Sol(); part.node_no_seq = [x for x in sol.node_no_seq if x not in remove_list]
    part.obj, part.route_list, part.route_distance = calObj(part.node_no_seq, model); return part

# ---------------------- 预计算kNN/全图（静态基边） ---------------------- #
def precompute_base_graph(node_list, model: Model, k_nn=10, use_full_graph_knn=False, weight_mode='inv'):
    n = len(node_list); edges = set()
    if use_full_graph_knn:
        for u in range(n):
            for v in range(n):
                if u != v: edges.add((u, v))
    else:
        k = min(k_nn, max(1, n - 1))
        for u in range(n):
            dists = [(v, model.distance_matrix[u, v]) for v in range(n) if v != u]
            dists.sort(key=lambda x: x[1])
            for v, _ in dists[:k]:
                edges.add((u, v)); edges.add((v, u))
    if not edges:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        ew = torch.empty((0,), dtype=torch.float, device=device); return ei, ew
    ei = torch.tensor(list(edges), dtype=torch.long, device=device).t().contiguous()
    if weight_mode == 'dist':
        ew = torch.tensor([float(model.distance_matrix[u, v]) for (u, v) in edges],
                          dtype=torch.float, device=device)
    else:
        ew = torch.tensor([1.0 / (float(model.distance_matrix[u, v]) + 1e-6) for (u, v) in edges],
                          dtype=torch.float, device=device)
    return ei, ew

# ---------------------- 构图：基边 + 路径边 + served_flag ---------------------- #
def build_graph(sol: Sol, node_list, model: Model, base_edge_index, base_edge_weight,
                removed_set=None, include_path_edges=True):
    n = len(node_list)
    served_flag = [1.0] * n
    if removed_set is not None:
        for rid in removed_set:
            if 0 <= rid < n: served_flag[rid] = 0.0
    x = torch.tensor([[node.x_coord, node.y_coord, node.demand, served_flag[i]]
                      for i, node in enumerate(node_list)], dtype=torch.float, device=device)

    edge_index = base_edge_index; edge_weight = base_edge_weight
    if include_path_edges and sol.route_list:
        dyn_edges = set()
        for route in sol.route_list:
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if 0 <= u < n and 0 <= v < n:
                    dyn_edges.add((u, v)); dyn_edges.add((v, u))
        if dyn_edges:
            ei2 = torch.tensor(list(dyn_edges), dtype=torch.long, device=device).t().contiguous()
            ew2 = torch.tensor([1.0/(model.distance_matrix[u, v] + 1e-6) for (u, v) in dyn_edges],
                               dtype=torch.float, device=device)
            edge_index = torch.cat([edge_index, ei2], dim=1)
            edge_weight = torch.cat([edge_weight, ew2], dim=0)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

# ---------------------- GNN 编码器 & Agent ---------------------- #
class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=32, use_attention=True):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        # 记录“每路分支”的维度，最终输出会是 2 * out_dim（Avg + Max）
        self.out_dim = out_dim

        # 可加权：用线性打分 -> softmax 作为注意力权重
        self.use_attention = use_attention
        self.score = torch.nn.Linear(out_dim, 1) if use_attention else None

        # 无边图/退化情况的兜底（与原先一致）
        self.fallback = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        w = getattr(data, 'edge_weight', None)

        if edge_index is None or edge_index.numel() == 0:
            h = self.fallback(x)                         # [N, d]
        else:
            h = F.relu(self.conv1(x, edge_index, w))     # [N, hid]
            h = self.conv2(h, edge_index, w)             # [N, d]

        # Avg（可加权）
        if self.use_attention:
            scores = self.score(h).squeeze(-1)           # [N]
            alpha = torch.softmax(scores, dim=0)         # [N]
            g_avg = (alpha.unsqueeze(-1) * h).sum(dim=0, keepdim=True)  # [1, d]
        else:
            g_avg = h.mean(dim=0, keepdim=True)          # [1, d]

        # Max（无权重，逐维最大）
        g_max, _ = h.max(dim=0, keepdim=True)            # [1, d]

        # 按论文拼接 Avg 与 Max
        return torch.cat([g_avg, g_max], dim=-1)         # [1, 2d]


class DACAgent(torch.nn.Module):
    def __init__(self, enc_dim=32, extra_state_dim=2, hid=32, gamma=0.9, ent_coef=0.01):
        super().__init__()
        # 注意：开启 Avg+Max（可加权）
        self.encoder = GraphEncoder(in_dim=4, hid=hid, out_dim=enc_dim, use_attention=True).to(device)
        enc_feat_dim = self.encoder.out_dim * 2  # Avg + Max 并行，维度翻倍

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1)
        ).to(device)

        self.actor_destroy = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 2)  # 2个破坏算子
        ).to(device)

        self.actor_repair = torch.nn.Sequential(
            torch.nn.Linear(enc_feat_dim + extra_state_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 3)  # 3个修复算子
        ).to(device)

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.opt_ad = torch.optim.Adam(self.actor_destroy.parameters(), lr=1e-3)
        self.opt_ar = torch.optim.Adam(self.actor_repair.parameters(), lr=1e-3)

        self.gamma = gamma; self.ent_coef = ent_coef
        # 破坏算子数量（用于 Destroy_id 归一化）；与 actor_destroy 的输出维保持一致
        self.n_destroy = 2

    # def build_state(self, graph: Data, ratio: float):
    #     g = self.encoder(graph)
    #     r = torch.tensor([[float(ratio)]], device=device, dtype=g.dtype)
    #     return torch.cat([g, r], dim=-1)

    def build_state(self, graph: Data, ratio: float, destroy_id: int = 0):
        """
        状态 = [图嵌入 g, ratio, destroy_id_norm]
        - ratio: 破坏比例；破坏前可设 0.0
        - destroy_id: 破坏算子编号（修复阶段传入 a_d）。按 0~1 归一化后再拼接
        """
        g = self.encoder(graph)
        r = torch.tensor([[float(ratio)]], device=device, dtype=g.dtype)

        if self.n_destroy > 1:
            d_norm = float(destroy_id) / float(self.n_destroy - 1)
        else:
            d_norm = 0.0
        d = torch.tensor([[d_norm]], device=device, dtype=g.dtype)

        return torch.cat([g, r, d], dim=-1)

    def _sample_with_eps(self, logits, eps, n_actions):
        if random.random() < eps:
            return random.randrange(n_actions)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs).sample().item()

    def act_destroy(self, state, eps=0.1):
        logits = self.actor_destroy(state); return self._sample_with_eps(logits, eps, 2)

    def act_repair(self, state, eps=0.1):
        logits = self.actor_repair(state);  return self._sample_with_eps(logits, eps, 3)

    def update(self, d_trans, r_trans):
        sd  = torch.cat(d_trans['states'], dim=0).to(device)
        ad  = torch.tensor(d_trans['actions'], dtype=torch.long, device=device).view(-1, 1)
        rd  = torch.tensor(d_trans['rewards'], dtype=torch.float, device=device).view(-1, 1)
        sdn = torch.cat(d_trans['next_states'], dim=0).to(device)
        dd  = torch.tensor(d_trans['dones'], dtype=torch.float, device=device).view(-1, 1)

        sr  = torch.cat(r_trans['states'], dim=0).to(device)
        ar  = torch.tensor(r_trans['actions'], dtype=torch.long, device=device).view(-1, 1)
        rr  = torch.tensor(r_trans['rewards'], dtype=torch.float, device=device).view(-1, 1)
        srn = torch.cat(r_trans['next_states'], dim=0).to(device)
        dr  = torch.tensor(r_trans['dones'], dtype=torch.float, device=device).view(-1, 1)

        with torch.no_grad():
            vdn = self.critic(sdn); vrn = self.critic(srn)
            td_tgt_d = rd + self.gamma * vdn * (1 - dd)
            td_tgt_r = rr + self.gamma * vrn * (1 - dr)

        v_d = self.critic(sd); v_r = self.critic(sr)
        critic_loss = F.mse_loss(v_d, td_tgt_d) + F.mse_loss(v_r, td_tgt_r)

        td_delta_d = (td_tgt_d - v_d).detach()
        td_delta_r = (td_tgt_r - v_r).detach()

        logits_d = self.actor_destroy(sd)
        logits_r = self.actor_repair(sr)
        logp_d = torch.log(F.softmax(logits_d, dim=1).gather(1, ad) + 1e-8)
        logp_r = torch.log(F.softmax(logits_r, dim=1).gather(1, ar) + 1e-8)

        # 熵正则以鼓励探索
        ent_d = -(F.softmax(logits_d, dim=1) * F.log_softmax(logits_d, dim=1)).sum(dim=1, keepdim=True).mean()
        ent_r = -(F.softmax(logits_r, dim=1) * F.log_softmax(logits_r, dim=1)).sum(dim=1, keepdim=True).mean()

        actor_d_loss = torch.mean(-logp_d * td_delta_d) - self.ent_coef * ent_d
        actor_r_loss = torch.mean(-logp_r * td_delta_r) - self.ent_coef * ent_r

        total_loss = critic_loss + actor_d_loss + actor_r_loss
        self.opt_enc.zero_grad(); self.opt_critic.zero_grad(); self.opt_ad.zero_grad(); self.opt_ar.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                       list(self.critic.parameters()) +
                                       list(self.actor_destroy.parameters()) +
                                       list(self.actor_repair.parameters()), max_norm=1.0)
        self.opt_enc.step(); self.opt_critic.step(); self.opt_ad.step(); self.opt_ar.step()

# ---------------------- 画图 ---------------------- #
def plotObj(obj_list):
    try:
        plt.figure(); plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
        plt.xlabel('Iterations'); plt.ylabel('Obj Value'); plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(); plt.show()
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
        plt.tight_layout(); plt.show()
    except Exception:
        pass

# ---------------------- CLI ---------------------- #
parser = argparse.ArgumentParser(description='DAC-ALNS with Dual-Stage TD (fixed)')
parser.add_argument('file_path', type=str, help='CSV文件路径')
parser.add_argument('result_file_path', type=str, help='结果聚合JSON所在目录的同级路径')
parser.add_argument('--knn_k', type=int, default=5, help='kNN邻居数')
parser.add_argument('--full_graph_knn', action='store_true', help='使用全连接图替代kNN')
parser.add_argument('--weight_mode', choices=['inv', 'dist'], default='inv', help='边权：inv(1/d)或dist(d)')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
parser.add_argument('--eps', type=float, default=0.10, help='epsilon-greedy 探索概率')
args = parser.parse_args()
set_seed(args.seed)

# ---------------------- 超参 ---------------------- #
v_cap = 100
rand_d_max, rand_d_min = 0.4, 0.1
worst_d_min, worst_d_max = 5, 20
regret_n = 2
r1, r2, r3 = 40, 10, -0.1
rho = 0.6
gamma = 0.9
T0 = 50.0
phi = 0.995
ent_coef = 0.01

# ---------------------- 初始化 ---------------------- #
model = Model()
model.vehicle_cap = v_cap
model.rand_d_max, model.rand_d_min = rand_d_max, rand_d_min
model.worst_d_min, model.worst_d_max = worst_d_min, worst_d_max
model.regret_n = regret_n
model.r1, model.r2, model.r3 = r1, r2, r3
model.rho = rho
model.T0 = T0
model.phi = phi

readCsvFile(args.file_path, model)
calDistanceMatrix(model)

# 索引==id 的 nodes
N = len(model.node_id_list) + 1
nodes = [None] * N
for nid, node in model.demand_dict.items():
    nodes[nid] = node

# 初解
sol = Sol()
sol.node_no_seq = genInitialSol(model.node_id_list)
sol.obj, sol.route_list, sol.route_distance = calObj(sol.node_no_seq, model)
model.best_sol = copy.deepcopy(sol)

# 预计算基边
base_edge_index, base_edge_weight = precompute_base_graph(
    nodes, model, k_nn=args.knn_k, use_full_graph_knn=args.full_graph_knn, weight_mode=args.weight_mode
)

# Agent
agent = DACAgent(enc_dim=64, extra_state_dim=2, hid=64, gamma=gamma, ent_coef=0.01).to(device)

history_best_obj = [sol.obj]
return_list = []
start_time = time.time()
num_steps = max(50 * model.number_of_demands, 2000)
all_iterations = []

# ---------------------- 主循环（双阶段TD） ---------------------- #
for it in range(num_steps):
    # 完整解 s
    g_s = build_graph(sol, nodes, model, base_edge_index, base_edge_weight, removed_set=None, include_path_edges=True)
    # s = agent.build_state(g_s, ratio=0.0)
    s = agent.build_state(g_s, ratio=0.0, destroy_id=0)

    # 破坏
    a_d = agent.act_destroy(s, eps=args.eps); model.d_select[a_d] += 1
    remove_list = doDestroy_by_id(a_d, model, sol)
    removed_set = set(remove_list); des_ratio = len(remove_list) / max(1, len(model.node_id_list))

    # 破坏后 s_d
    part_sol = apply_destroy_to_partial_sol(sol, remove_list, model)
    g_sd = build_graph(part_sol, nodes, model, base_edge_index, base_edge_weight, removed_set=removed_set, include_path_edges=True)
    s_d = agent.build_state(g_sd, ratio=des_ratio, destroy_id=a_d)

    # 修复
    a_r = agent.act_repair(s_d, eps=args.eps); model.r_select[a_r] += 1
    new_sol = doRepair_by_id(a_r, remove_list, model, sol)

    # s'
    g_sprime = build_graph(new_sol, nodes, model, base_edge_index, base_edge_weight, removed_set=None, include_path_edges=True)
    # 破坏后状态（供修复 Actor 使用）：传入本轮破坏动作 a_d
    s_prime = agent.build_state(g_sprime, ratio=des_ratio, destroy_id=a_d)

    # ---- 破坏后（已有 part_sol, s_d）----
    C_best, C_before, C_destroy = model.best_sol.obj, sol.obj, part_sol.obj
    impr_ratio_d = (C_best - C_destroy) / max(1e-6, C_best)
    if C_best < C_destroy < C_before:
        d_reward = model.alpha1 + model.beta * impr_ratio_d
    elif C_destroy <= C_best:
        d_reward = model.alpha2 + model.beta * impr_ratio_d
    else:
        d_reward = 0.0  # 破坏导致明显变差，不给奖

    # ---- 修复（已有 new_sol）----
    C_repair = new_sol.obj
    f_rep_id = model.r_select[a_r]
    f_max = max(1.0, model.r_select.max())
    explore_bonus = model.delta * (1.0 - f_rep_id / f_max)
    if C_repair < C_best:
        r_reward = model.r1 + explore_bonus
    elif C_best <= C_repair < C_before:
        r_reward = model.r2 + explore_bonus
    else:
        r_reward = model.r3 + explore_bonus

    # === 接纳 / 刷新（关键）===
    accepted = False
    if C_repair < C_before:
        sol = copy.deepcopy(new_sol)
        accepted = True
        if C_repair < model.best_sol.obj:
            model.best_sol = copy.deepcopy(new_sol)
    elif math.exp((C_before - C_repair) / max(model.T0, 1e-3)) >= random.random():
        sol = copy.deepcopy(new_sol)
        accepted = True

    # 让 r_trans 的 next_state 更贴近“真实环境”的下一状态（虽然 done=1 不会用到）
    g_next = g_sprime if accepted else g_s
    # s_next = agent.build_state(g_next, ratio=des_ratio)
    s_next = agent.build_state(g_next, ratio=des_ratio, destroy_id=a_d)

    # Double-stage transitions
    d_trans = {'states': [s], 'actions': [a_d], 'rewards': [d_reward], 'next_states': [s_d], 'dones': [0.0]}
    r_trans = {'states': [s_d], 'actions': [a_r], 'rewards': [r_reward], 'next_states': [s_next], 'dones': [1.0]}

    # 更新
    model.T0 *= model.phi
    return_list.append(d_reward)
    return_list.append(r_reward)
    agent.update(d_trans, r_trans)

    # 日志
    run_time = time.time() - start_time
    history_best_obj.append(model.best_sol.obj)
    if (it + 1) % 100 == 0:
        print(f"{Path(args.file_path).name} | Iter {it + 1}/{num_steps} | Best Obj: {model.best_sol.obj:.2f} | Time: {run_time:.2f}s")
    if (it + 1) % 10 == 0:
        all_iterations.append({
            "iteration": it + 1,
            "file_path": args.file_path,
            "best_obj": round(model.best_sol.obj, 2),
            "run_time": round(run_time, 2),
            "rand_des": float(model.d_select[0]),
            "wors_des": float(model.d_select[1]),
            "rand_rep": float(model.r_select[0]),
            "gree_rep": float(model.r_select[1]),
            "regr_rep": float(model.r_select[2]),
        })

# ---------------------- 收尾 ---------------------- #
parent_path = Path(args.result_file_path).parent
parts = Path(args.result_file_path).parts
file_name = Path(args.result_file_path).stem

log_dir = parent_path / 'LOG'; log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / f'{file_name}_iteration_log_{round(model.best_sol.obj, 2)}_{round(run_time, 2)}.json'
with open(log_file_path, 'w', encoding='utf-8') as f:
    json.dump(all_iterations, f, indent=4, ensure_ascii=False)

try:
    result_path = parent_path / f'{parts[-2]}_result.json'
    safe_append_to_json(result_path, file_name, round(model.best_sol.obj, 2), round(run_time, 2))
except IndexError:
    print("Warn: result aggregation path not found; skip.")

try:
    model_dir = parent_path.parent.parent / 'SavedModel' / parts[-2]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f'{file_name}_model_{round(model.best_sol.obj, 2)}.pth'
    torch.save(agent.state_dict(), model_save_path)
except IndexError:
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

# 可选：本地可视化
try:
    plt.figure(); plt.plot(np.arange(1, len(history_best_obj) + 1), history_best_obj)
    plt.xlabel('Iterations'); plt.ylabel('Best Obj'); plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout(); plt.close()
except Exception:
    pass
