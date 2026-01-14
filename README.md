# DAC-ALNS: Dual-Stage Actor-Critic Adaptive Large Neighborhood Search

本项目包含一系列基于 **深度强化学习 (Deep Reinforcement Learning)** 与 **自适应大邻域搜索 (Adaptive Large Neighborhood Search, ALNS)** 结合的车辆路径问题 (VRP) 求解程序。

## 项目简介

本项目实现了 **DAC-ALNS (Dual-Stage Actor-Critic ALNS)** 算法框架。该框架通过图神经网络 (GNN) 提取路径问题的空间特征，并利用双阶段 Actor-Critic 架构在搜索过程中动态选择最优的破坏 (Destroy) 和修复 (Repair) 算子。

### 核心特性

- **双阶段 Actor-Critic 架构**：分别为破坏和修复阶段设计独立的 Actor 网络，实现精细化的算子选择。
- **图神经网络 (GNN) 编码器**：使用 `GCNConv` 提取节点特征和路径拓扑信息，支持时间窗、需求量等多种特征。
- **自适应搜索机制**：结合了 ALNS 的探索能力与强化学习的决策能力。
- **高性能计算支持**：利用 `Numba` 加速核心计算逻辑（如 Split 算法和时刻表计算），支持 GPU 加速。

## 程序说明

本仓库目前包含以下两个核心程序：

### 1. dac-alns-cvrp.py
针对 **带容量限制的车辆路径问题 (CVRP)** 的求解程序。
- **特点**：侧重于容量约束下的路径优化。
- **运行方式**：
  ```bash
  python dac-alns-cvrp.py <csv_path> <result_file_path> [options]
  ```

### 2. dac-alns-vrptw.py
针对 **带时间窗和容量限制的车辆路径问题 (VRPTW)** 的求解程序。
- **特点**：在 CVRP 的基础上引入了严格的时间窗约束（Ready Time, Due Time, Service Time）。
- **状态编码**：节点特征显式加入了时间窗影响（如到期紧迫度、最早到达等待等）。
- **运行方式**：
  ```bash
  python dac-alns-vrptw.py <csv_path> <result_file_path> [options]
  ```

## 快速开始

### 环境依赖

- Python 3.8+
- PyTorch
- torch_geometric
- NumPy
- Matplotlib
- Numba (强烈建议安装以获得显著加速)

### 通用参数说明
- `csv_path`: 输入的实例文件路径（CSV 格式）。
- `result_file_path`: 结果保存的 JSON 文件路径。
- `--knn_k`: kNN 图的邻居数量。
- `--weight_mode`: 边权重模式，可选 `inv` (1/距离) 或 `dist` (距离)。
- `--eps`: Epsilon-Greedy 探索概率。

## 许可证

本项目仅供学习与研究使用。
