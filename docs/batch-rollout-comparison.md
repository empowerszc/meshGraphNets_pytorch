# Batch Rollout 实现方式对比

**日期**: 2026-04-20  
**目的**: 掌握"不规则数据"的批处理技巧

---

## 核心挑战

Rollout 推理的"不规则"特性：

| 维度 | 特性 | 原因 |
|------|------|------|
| **时间** | 必须串行 | 第 t 步依赖 t-1 步的预测值（自回归） |
| **轨迹** | 可以并行 | 不同轨迹完全独立 |
| **图结构** | 特殊处理 | PyG 的 Batch 是"合并图"而非"数据并行" |

---

## 四种实现方式

### V1 - 手动索引（基准版本）

**文件**: `batch_rollout.py`

**核心思路**:
```python
# 按轨迹组织索引
trajectories_indices = [
    [0, 1, 2, ..., 598],       # 轨迹 0 的 599 步
    [599, 600, ..., 1197],     # 轨迹 1 的 599 步
    ...
]

# 按时间步循环（串行）
for step in range(num_steps):
    # 收集所有轨迹的当前步
    step_graphs = [dataset[tra_idx * num_steps + step] for tra_idx in range(num_samples)]
    
    # 分 batch 并行处理
    for batch_start in range(0, num_samples, batch_size):
        batch_graphs = step_graphs[batch_start:batch_start+batch_size]
        batched = Batch.from_data_list(batch_graphs)
        pred = model(batched)  # 一次 forward 处理多条轨迹
```

**优点**:
- 最直接，易于理解
- 无额外抽象层
- 内存效率高

**缺点**:
- 索引计算分散在代码中
- 不易复用

**适用场景**: 快速原型、学习理解

---

### V2 - Subset + DataLoader

**文件**: `batch_rollout_v2.py`

**核心思路**:
```python
# 为每个时间步创建 Subset
step_subsets = []
for step in range(num_steps):
    # 该时间步在所有轨迹中的全局索引
    indices = [tra_idx * num_steps + step for tra_idx in range(num_samples)]
    step_subsets.append(Subset(dataset, indices))

# 按时间步循环
for step in range(num_steps):
    # 使用 DataLoader 加载当前时间步
    loader = DataLoader(step_subsets[step], batch_size=batch_size)
    for batch_graphs in loader:
        # 处理 batch...
```

**关键 API**:
- `Subset(dataset, indices)`: 创建数据集的"视图"
- `DataLoader`: 支持 `num_workers` 多进程加载

**优点**:
- 使用标准 PyTorch API
- 支持多进程数据加载
- 代码结构清晰

**缺点**:
- 需要为每个时间步创建 Subset（599 个）
- 索引计算仍较复杂

**适用场景**: 需要多进程加载数据时

---

### V3 - 自定义 TrajectoryDataset

**文件**: `batch_rollout_v3.py`

**核心思路**:
```python
class TrajectoryDataset(Dataset):
    """按轨迹组织的 Dataset 包装器"""
    
    def __getitem__(self, tra_idx) -> List[Data]:
        """返回整条轨迹的所有时间步"""
        return [
            dataset[tra_idx * num_steps + step]
            for step in range(num_steps)
        ]

# 使用
traj_dataset = TrajectoryDataset(dataset, num_samples)
loader = DataLoader(traj_dataset, batch_size=batch_size)

# 一次性获取所有轨迹
all_trajectories = []
for batch in loader:
    all_trajectories.extend(batch)

# 按时间步处理
for step in range(num_steps):
    step_graphs = [traj[step] for traj in all_trajectories]
```

**优点**:
- 封装良好，接口清晰
- 可按轨迹并行加载数据
- 易于扩展（如添加数据增强）

**缺点**:
- 需要一次性获取所有轨迹到内存
- DataLoader 的 collate_fn 不能是生成器

**适用场景**: 需要按轨迹组织数据的场景

---

### V4 - 3D 张量预加载

**文件**: `batch_rollout_v4.py`

**核心思路**:
```python
# 一次性加载所有数据
all_velocities = []  # List[[num_steps, N_i, 2]]
all_positions = []   # List[[N_i, 2]]
all_cells = []       # List[[F_i, 3]]

for tra_idx in range(num_samples):
    velocities = []
    for step in range(num_steps):
        graph = dataset[start_idx + step]
        velocities.append(graph.y.clone())
    all_velocities.append(torch.stack(velocities))

# 按时间步切片
for step in range(num_steps):
    step_velocities = [vel[step] for vel in all_velocities]
    # 构建图 → 打包 → 预测
```

**优点**:
- 代码最简洁
- 数据访问最快（内存连续）
- 无需反复读取 dataset

**缺点**:
- 内存占用大（不适合大数据集）
- 不支持在线数据增强

**适用场景**: 小数据集、追求简洁代码

---

## 性能对比

| 版本 | 2 条轨迹耗时 | 代码行数 | 内存占用 | 可读性 |
|------|-------------|---------|---------|--------|
| V1 | ~135s | 280 | 低 | ⭐⭐⭐⭐ |
| V2 | ~134s | 250 | 低 | ⭐⭐⭐⭐⭐ |
| V3 | ~129s | 290 | 中 | ⭐⭐⭐⭐ |
| V4 | ~140s | 240 | 高 | ⭐⭐⭐⭐⭐ |

**注意**: 所有版本的推理逻辑相同，耗时差异主要来自 overhead

---

## 关键 API 总结

### PyTorch 基础
```python
from torch.utils.data import Dataset, DataLoader, Subset

# 自定义 Dataset
class MyDataset(Dataset):
    def __getitem__(self, idx): ...
    def __len__(self): ...

# Subset - 创建数据集视图
subset = Subset(dataset, [0, 10, 20, 30])

# DataLoader - 批处理
loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
```

### PyG Batch
```python
from torch_geometric.data import Batch

# 打包多个图
batched = Batch.from_data_list([graph1, graph2, graph3])

# 拆分结果
graphs = [graph1, graph2, graph3]
num_nodes_list = [g.num_nodes for g in graphs]
split_results = torch.split(batched_output, num_nodes_list, dim=0)
```

---

## 选择建议

| 需求 | 推荐版本 |
|------|---------|
| 快速原型/学习 | V1 |
| 需要多进程加载 | V2 |
| 按轨迹组织数据 | V3 |
| 小数据集/简洁 | V4 |
| 生产环境 | V1 或 V3（平衡性能和可维护性） |

---

## 学习路径

1. **先理解 V1** - 掌握核心逻辑（时间串行、轨迹并行）
2. **学习 V2** - 理解 Subset 和 DataLoader 的用法
3. **学习 V3** - 学习自定义 Dataset 的封装
4. **学习 V4** - 理解内存 vs 速度的权衡

---

## 扩展思考

1. **如果数据集有 1000 条轨迹** - V4 内存不足，用 V1/V2/V3
2. **如果需要数据增强** - V3 最容易扩展（在 `__getitem__` 中增强）
3. **如果图结构完全相同** - 可进一步优化为真正的 4D 张量 `[B, T, N, F]`
4. **如果需要分布式推理** - V2/V3 最容易适配 DistributedDataLoader
