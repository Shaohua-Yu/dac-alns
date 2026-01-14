# DAC-ALNS for CVRP (Capacitated Vehicle Routing Problem)

这是一个基于 **深度强化学习 (Deep Reinforcement Learning)** 与 **自适应大邻域搜索 (Adaptive Large Neighborhood Search, ALNS)** 结合的带容量限制的车辆路径问题 (CVRP) 求解程序。

## 项目简介

本项目实现了一种名为 **DAC-ALNS (Dual-Stage Actor-Critic ALNS)** 的算法。该算法通过图神经网络 (GNN) 提取车辆路径问题的空间特征，并利用 Actor-Critic 框架在搜索过程中动态选择最优的破坏 (Destroy) 和修复 (Repair) 算子，从而实现对 CVRP 问题的高效求解。

### 核心特性

- **双阶段 Actor-Critic 架构**：分别针对破坏和修复阶段设计 Actor 网络，实现精细化的算子选择策略。
- **图神经网络 (GNN) 编码器**：使用 `GCNConv` 提取节点特征和路径拓扑信息，支持 kNN 邻居图和全连接图模式。
- **自适应搜索机制**：结合了 ALNS 的探索能力与强化学习的决策能力，能够根据当前解的状态动态调整搜索方向。
- **熵正则化与探索**：引入熵正则化项和 Epsilon-Greedy 策略，确保算法在搜索过程中保持足够的探索度，避免陷入局部最优。
- **高性能计算支持**：利用 `Numba` 加速核心计算逻辑（如 Split 算法），并支持 GPU 加速。

## 算法流程

1. **状态编码**：将当前解转换为图数据结构，通过 GNN 编码器生成全局状态嵌入。
2. **破坏阶段**：Destroy Actor 根据当前状态选择破坏算子（如随机破坏、最差破坏），移除部分节点。
3. **修复阶段**：Repair Actor 根据破坏后的残解状态选择修复算子（如随机修复、贪婪修复、后悔修复），重新插入节点。
4. **策略更新**：基于解的质量改进情况（奖励信号），通过双阶段 TD 误差更新网络参数。

## 快速开始

### 环境依赖

- Python 3.8+
- PyTorch
- torch_geometric
- NumPy
- Matplotlib
- Numba (可选，用于加速)

### 运行示例

使用以下命令运行程序：

```bash
python Duel_AC_CVRP_v1.py <csv_path> <result_file_path> [--knn_k 10] [--full_graph_knn] [--weight_mode inv|dist] [--seed 0]
```

#### 参数说明：
- `csv_path`: 输入的 CVRP 实例文件路径（CSV 格式）。
- `result_file_path`: 结果保存的 JSON 文件路径。
- `--knn_k`: kNN 图的邻居数量（默认为 5）。
- `--full_graph_knn`: 是否使用全连接图。
- `--weight_mode`: 边权重模式，可选 `inv` (1/距离) 或 `dist` (距离)。

## 文件说明

- `Duel_AC_CVRP_v1.py`: 主程序文件，包含模型定义、训练循环及推理逻辑。

## 许可证

本项目仅供学习与研究使用。
