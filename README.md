# DAC-ALNS

DAC-ALNS 是一个面向车辆路径问题（Vehicle Routing Problem, VRP）的研究与复现实验代码仓库，包含自适应大邻域搜索（ALNS）、Actor-Critic、DAC 变体以及若干外部基线方法的源码、测试实例、配置和统计结果。

当前发布包以可复现的精简源码和数据为主。主入口是 `run.py`；历史实现保存在 `original_source/`，参考实验脚本保存在 `reference_code/`。

## 支持的问题

- **CVRP**：带容量约束的车辆路径问题。
- **VRPTW**：带容量约束和时间窗的车辆路径问题。

数据集包括 Augerat A/B/P、CVRPLIB X 系列 CVRP 实例，以及 Solomon 100-customer 的 C、R、RC 系列 VRPTW 实例。处理后的 CSV 输入位于 `instances/`，实例路径和 SHA-256 校验值记录在 `configs/instances.json` 中。

## 快速开始

本项目要求 Python 3.10。建议在虚拟环境中安装依赖：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖包括 NumPy、PyTorch、PyTorch Geometric、Matplotlib、Numba；统计参考脚本还使用 SciPy。完整版本固定在 `requirements.txt` 中。

查看命令行参数：

```bash
python run.py --help
```

运行一个独立的 CVRP 试验（100 次迭代）：

```bash
python run.py \
  --problem CVRP \
  --instance A-n32-k5 \
  --method DAC-Partial \
  --iterations 100
```

运行一个独立的 VRPTW 试验（10 秒）：

```bash
python run.py \
  --problem VRPTW \
  --instance C101 \
  --method ALNS \
  --seconds 10
```

可用方法为：

- `ALNS`
- `AC`
- `DAC-Partial`
- `DAC-Pre`
- `DAC-Credit`

其他常用参数：

- `--instance`：实例 ID，大小写不敏感，例如 `A-n32-k5` 或 `C101`；
- `--seed`：随机种子，默认值为 `0`；
- `--iterations`：正整数迭代预算；
- `--seconds`：正数时间预算，与 `--iterations` 互斥；
- `--checkpoint`：可选的兼容模型检查点路径。

`run.py` 默认使用 CPU，并将独立运行结果写入 `reproduced/`。该目录及运行时缓存不会纳入版本控制。若使用神经方法但不提供检查点，程序会从随机权重开始；这与正式实验中使用的预训练检查点并不等价。

## 目录说明

```text
run.py                         独立 CPU 试验入口
verify.py                      发布包完整性校验
requirements.txt               Python 依赖及版本
configs/
  instances.json               实例清单、路径和哈希
  experiments.json             实验协议、预算和汇总信息
  method_settings.json         方法与实验设置
  training_jobs/               训练任务配置
instances/
  CVRP/                        CVRP CSV 实例
  VRPTW/                       VRPTW CSV 实例
src/
  controlled_engine.py         统一试验引擎
  experiment_io.py             实验输入输出工具
  validate_routes.py            路线和目标值验证工具
original_source/               原始 DAC/AC 实现
reference_code/                参考、迁移、训练和统计脚本
results/                       已归档的实验结果 CSV
sources.json                   文件来源与哈希记录
UPSTREAMS.txt                  外部基线和数据来源说明
checksums.json                 发布文件完整性清单
```

## 结果与实验协议

`results/` 保存已完成实验的 CSV 结果，按实验协议分为：

- `full_revision/`：完整修订实验结果；
- `equal_iterations/`：等迭代预算比较结果；
- `time_only/`：仅按时间预算停止的结果。

实验配置位于 `configs/experiments.json` 和 `configs/method_settings.json`。正式实验采用 CPU、单线程和未四舍五入的欧氏距离，目标主要是总距离；车队规模在该协议中不作为绑定目标。正式 VRPTW 神经实验使用过预训练检查点，但这些检查点不包含在本精简发布包中。

`reference_code/` 中部分外部适配器依赖单独提供的上游实现、检查点或额外环境，不能仅凭本仓库直接运行。相关来源、记录提交和许可说明见 `UPSTREAMS.txt` 与 `sources.json`。

## 完整性与开发检查

验证发布包中的文件是否与 `checksums.json` 一致：

```bash
python verify.py
```

成功时会输出包含 `"valid": true` 的 JSON。提交代码前也可以运行 Python 语法检查：

```bash
python -m compileall -q -f run.py verify.py src original_source reference_code
```

## 数据与来源

仓库中的源码来源、转换关系及文件哈希记录在 `sources.json` 中。外部基线 DR-ALNS 和 PPO-ALNS 的来源及记录提交见 `UPSTREAMS.txt`。上游项目的代码、模型、依赖和许可条款仍由各自项目负责，本仓库不自动包含这些外部资源。

## 许可与使用说明

本仓库主要用于研究复现和实验分析。使用 `original_source/`、`reference_code/` 或外部适配器时，请同时遵守对应来源项目的许可和使用条件，并根据 `sources.json`、`UPSTREAMS.txt` 保留来源信息。
