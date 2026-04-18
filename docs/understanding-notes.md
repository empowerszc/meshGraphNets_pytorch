# MeshGraphNets PyTorch 实现 - 详细代码理解笔记

## 📋 目录

1. [项目概述](#1-项目概述)
2. [代码结构](#2-代码结构)
3. [核心组件详解](#3-核心组件详解)
4. [数据流与训练流程](#4-数据流与训练流程)
5. [关键技术点](#5-关键技术点)
6. [输入数据特点与 Rollout 详解](#6-输入数据特点与-rollout-详解)
7. [总结](#7-总结)
8. [模型结构深度解析](#8-模型结构深度解析)
9. [示例 Case 集合](#9-示例-case-集合)
10. [常见问题与调试技巧](#10-常见问题与调试技巧)
11. [ONNX 导出](#11-onnx-导出)
12. [参考资源](#12-参考资源)

---

**最新更新**:
- 2026-04-18: 添加 ONNX 导出指南、论文解读文档链接
- 修复 demo_inference.py rollout 崩溃问题

---

## 1. 项目概述

### 1.1 项目背景

本项目是 DeepMind 的 **MeshGraphNets** 论文的 PyTorch + PyG (PyTorch Geometric) 实现，用于学习基于网格的物理模拟。

**论文**: [Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)](https://arxiv.org/abs/2010.03409)

### 1.2 应用场景

- **主要任务**: 圆柱绕流 (flow around a circular cylinder) 的流体动力学模拟
- **核心优势**: 
  - 比传统求解器快 **10-100 倍**
  - 保持物理一致性
  - 可扩展到新的 PDE、材料或领域

### 1.3 方法核心思想

```
输入网格 → 图神经网络 → 预测加速度 → 积分得到下一时刻速度
```

将网格视为图结构，利用图神经网络学习物理演化规律。

---

## 2. 代码结构

```
meshGraphNets_pytorch/
├── dataset/
│   ├── __init__.py
│   └── fpc.py              # 圆柱绕流数据集加载
├── model/
│   ├── __init__.py
│   ├── blocks.py           # 基础构建块 (EdgeBlock, NodeBlock, ONNX 兼容版本)
│   ├── model.py            # Encoder-Processor-Decoder 架构 (含 ONNX 导出支持)
│   └── simulator.py        # 主模拟器模型
├── utils/
│   ├── __init__.py
│   ├── noise.py            # 噪声注入 (训练技巧)
│   ├── normalization.py    # 特征归一化
│   └── utils.py            # 工具类 (节点类型枚举)
├── docs/
│   ├── understanding-notes.md           # 详细代码理解笔记
│   ├── data-structure-explained.md      # 数据结构详解
│   ├── dataset-analysis.md              # 数据集详细分析
│   ├── onnx-export-guide.md             # ONNX 导出指南
│   ├── onnx-export-summary.md           # ONNX 导出总结
│   └── meshgraphnets-paper-explained.md # 论文入门解读
├── parse_tfrecord.py       # TensorFlow 数据解析
├── train.py                # 单 GPU 训练
├── train_ddp.py            # 多 GPU 分布式训练 (DDP)
├── rollout.py              # 长时序推理
├── render_results.py       # 结果可视化
├── demo_inference.py       # 快速推理演示 (支持随机权重/checkpoint)
├── export_onnx.py          # ONNX 模型导出
├── test_onnx.py            # ONNX 模型测试
├── requirements.txt
└── README.md
```

---

## 3. 核心组件详解

### 3.1 数据处理 (`dataset/fpc.py`)

#### FpcDataset 类

```python
class FpcDataset(Dataset):
    def __init__(self, data_root, split):
        # 加载元数据 (位置、节点类型、网格连接关系)
        self.meta = np.load(meta_path)
        # 使用 memmap 高效加载速度场数据
        self.fp = np.memmap(data_path, dtype='float32', mode='r', shape=shape)
    
    def __getitem__(self, index):
        # 构造 PyG 的 Data 对象
        graph = Data(x=x, pos=pos, face=face, y=y)
        return graph
```

**关键设计**:
- **memmap**: 避免将整个数据集加载到内存
- **索引计算**: 通过 `indices` 和 `cindices` 快速定位样本
- **输出格式**: `x = [node_type, velocity]` (11 维特征)

### 3.2 图构建块 (`model/blocks.py`)

#### EdgeBlock

```python
class EdgeBlock(nn.Module):
    def forward(self, graph):
        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]
        collected_edges = torch.cat([senders_attr, receivers_attr, edge_attr], dim=1)
        edge_attr = self.net(collected_edges)  # MLP 更新
```

**功能**: 基于发送者/接收者节点特征更新边特征

#### NodeBlock

```python
class NodeBlock(nn.Module):
    def forward(self, graph):
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0)
        collected_nodes = torch.cat([graph.x, agg_received_edges], dim=-1)
        x = self.net(collected_nodes)
```

**功能**: 聚合入边信息并更新节点特征

**关键点**: 使用 `torch_scatter.scatter_add` 实现高效的图聚合

### 3.3 编码器 - 处理器 - 解码器 (`model/model.py`)

#### 整体架构

```
Encoder → [GnBlock × 15] → Decoder
```

#### Encoder

```python
class Encoder(nn.Module):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
```

**作用**: 将原始特征映射到统一的隐空间 (128 维)

#### GnBlock (Graph Network Block)

```python
class GnBlock(nn.Module):
    def forward(self, graph):
        graph = self.eb_module(graph)   # 边更新
        graph = self.nb_module(graph)   # 节点更新
        # 残差连接
        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr
```

**关键设计**:
- **残差连接**: 缓解深层网络的梯度消失
- **顺序处理**: 先边更新，后节点更新

#### Decoder

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size)
    
    def forward(self, graph):
        return self.decode_module(graph.x)
```

**输出**: 2D 加速度预测

#### MLP 构建函数

```python
def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(
        nn.Linear(in_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return nn.Sequential(module, nn.LayerNorm(out_size))
```

### 3.4 模拟器 (`model/simulator.py`)

#### Simulator 类

```python
class Simulator(nn.Module):
    def __init__(self, message_passing_num=15, node_input_size=11, edge_input_size=3, device='cuda'):
        self.model = EncoderProcesserDecoder(...)
        self._output_normalizer = Normalizer(size=2, ...)      # 加速度归一化
        self._node_normalizer = Normalizer(size=11, ...)       # 节点特征归一化
        self.edge_normalizer = Normalizer(size=3, ...)         # 边特征归一化
```

#### 节点特征构造

```python
def update_node_attr(self, frames, types):
    node_type = types.squeeze(-1).long()
    one_hot = F.one_hot(node_type, num_classes=9)  # [N, 9]
    node_feats = torch.cat([frames, one_hot], dim=-1)  # [N, 11]
    return self._node_normalizer(node_feats, self.training)
```

**特征组成**: `velocity(2) + node_type_onehot(9) = 11`

#### 训练模式 vs 推理模式

```python
def forward(self, graph, velocity_sequence_noise):
    if self.training:
        # 训练：注入噪声 → 预测归一化加速度
        noised_frames = frames + velocity_sequence_noise
        predicted_acc_norm, target_acc_norm = model(...), normalize(target_acc)
        return predicted_acc_norm, target_acc_norm
    else:
        # 推理：清洁速度 → 预测并反归一化
        # 【关键修复】保存原始 graph.x，推理后恢复
        # 原因：多步 rollout 时，graph.x 会被覆盖导致 node_type 丢失
        original_x = graph.x.clone()
        
        predicted_acc_norm = model(graph)
        acc_update = self._output_normalizer.inverse(predicted_acc_norm)
        predicted_velocity = frames + acc_update
        
        # 恢复原始 graph.x（保留 node_type 用于下一次 rollout）
        graph.x = original_x
        
        return predicted_velocity
```

**关键差异**:
- 训练时输出 **归一化的加速度** 用于 loss 计算
- 推理时输出 **下一时刻的速度**

**修复说明** (2026-04-18):
- **问题**: 之前推理模式下 `graph.x = node_attr` 覆盖了原始数据，导致下一次 rollout 时 `node_type` 变成归一化后的特征值（最大达到 1.48 亿！）
- **解决**: 保存并恢复原始 `graph.x`，确保每次 rollout 都能正确读取 node_type
- **测试**: `python demo_inference.py --mode random --rollout_steps 50`

---

## 4. 数据流与训练流程

### 4.1 数据预处理流程

```
TFRecord (DeepMind 格式)
    ↓ parse_tfrecord.py
.npz (元数据) + .dat (速度场 memmap)
    ↓ FpcDataset
PyG Data 对象
    ↓ T.Compose 变换
    - FaceToEdge: 面 → 边
    - Cartesian: 相对坐标作为边特征
    - Distance: 距离作为边特征
```

### 4.2 训练循环 (`train.py`)

```python
def train_one_epoch(model, dataloader, optimizer, transformer, device, noise_std):
    for graph in dataloader:
        graph = transformer(graph).to(device)
        velocity_sequence_noise = get_velocity_noise(graph, noise_std)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        
        # 仅在 NORMAL 和 OUTFLOW 节点上计算 loss
        mask = (node_type == NORMAL) | (node_type == OUTFLOW)
        loss = MSE(predicted_acc[mask], target_acc[mask])
        
        loss.backward()
        optimizer.step()
```

### 4.3 多 GPU 分布式训练 (`train_ddp.py`)

```python
# 使用 DDP 包装模型
simulator = DDP(simulator, device_ids=[local_rank])

# 使用 DistributedSampler
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(dataset, sampler=train_sampler, ...)

# 分布式 loss 聚合
dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
```

### 4.4 长时序推理 (`rollout.py`)

```python
@torch.no_grad()
def rollout(model, dataset):
    for i in range(num_sampes_per_tra):
        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity  # 自回归更新
        
        predicted_velocity = model(graph, None)
        
        # 边界条件约束
        predicted_velocity[mask] = next_v[mask]  # 固定边界/入口速度
        
        predicteds.append(predicted_velocity)
        targets.append(next_v)
```

**关键点**:
- **自回归预测**: 用上一时刻预测作为下一时刻输入
- **边界约束**: 强制边界条件保持一致

---

## 5. 关键技术点

### 5.1 归一化 (`utils/normalization.py`)

#### 在线统计累积

```python
class Normalizer(nn.Module):
    def _accumulate(self, batched_data):
        self._acc_sum += sum(data)
        self._acc_sum_squared += sum(data^2)
        self._acc_count += count
    
    def _mean(self):
        return self._acc_sum / max(self._acc_count, 1)
    
    def _std_with_epsilon(self):
        variance = self._acc_sum_squared / count - mean^2
        std = sqrt(clamp(variance, min=0))
        return max(std, epsilon)
```

**设计亮点**:
- **累积统计**: 避免预先扫描整个数据集
- **数值稳定**: `epsilon` 防止除零，`clamp` 防止负方差
- **训练/推理一致**: `accumulate=True` 仅在训练时更新统计

### 5.2 噪声注入 (`utils/noise.py`)

```python
def get_velocity_noise(graph, noise_std):
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity.shape)
    mask = type != NodeType.NORMAL
    noise[mask] = 0  # 仅在 NORMAL 节点上加噪
    return noise
```

**目的**:
- 提高模型鲁棒性
- 模拟数值误差
- 仅在流体区域加噪，边界条件保持干净

### 5.3 节点类型 (`utils/utils.py`)

```python
class NodeType(enum.IntEnum):
    NORMAL = 0       # 普通流体节点
    OBSTACLE = 1     # 障碍物
    AIRFOIL = 2      # 翼型
    HANDLE = 3       # 控制点
    INFLOW = 4       # 入口
    OUTFLOW = 5      # 出口
    WALL_BOUNDARY = 6  # 壁面
    SIZE = 9
```

### 5.4 边特征构造

```python
transformer = T.Compose([
    T.FaceToEdge(),      # 从三角面生成边
    T.Cartesian(norm=False),  # 相对位置 (dx, dy)
    T.Distance(norm=False)    # 距离 ||dx||
])
```

**边特征维度**: `3 = 2(相对坐标) + 1(距离)`

---

## 6. 输入数据特点与 Rollout 详解

### 6.1 模型输入数据的特征

#### 节点特征 (Node Features) - `graph.x`

```
维度：[N, 11]  N = 网格节点数量

特征组成:
┌────────────────────────────────────────────────────────┐
│  velocity (2 维)  │  node_type_onehot (9 维)            │
│  [vx, vy]        │  [0,0,1,0,0,0,0,0,0]               │
└────────────────────────────────────────────────────────┘
         ↓                        ↓
   当前速度场              节点类型独热编码
```

**node_type 的 9 种类型**:

| 值 | 类型 | 说明 | 是否预测 |
|---|---|---|---|
| 0 | NORMAL | 普通流体区域 | ✅ 预测 |
| 1 | OBSTACLE | 障碍物 | ❌ 固定 |
| 2 | AIRFOIL | 翼型表面 | ❌ 固定 |
| 3 | HANDLE | 控制点 | ❌ 固定 |
| 4 | INFLOW | 入口边界 | ❌ 固定 |
| 5 | OUTFLOW | 出口边界 | ✅ 预测 |
| 6 | WALL_BOUNDARY | 壁面 | ❌ 固定 |

**关键特点**:
- **混合特征**: 连续值 (速度) + 离散类型 (one-hot)
- **选择性预测**: 只在 NORMAL 和 OUTFLOW 节点上计算 loss
- **边界条件固定**: INFLOW、WALL 等节点的速度在推理时被强制重置

#### 边特征 (Edge Features) - `graph.edge_attr`

```python
# 通过 PyG 变换生成
transformer = T.Compose([
    T.FaceToEdge(),      # 从三角网格生成边
    T.Cartesian(norm=False),  # 相对坐标 (dx, dy)
    T.Distance(norm=False)    # 距离 ||dx||
])
```

```
维度：[E, 3]  E = 边的数量

特征组成:
┌─────────────────────────────────┐
│  dx    │    dy    │   distance  │
│  0.01  │   0.02   │    0.022    │
└─────────────────────────────────┘
     相对坐标              标量距离
```

**关键特点**:
- **几何信息**: 编码网格的空间结构
- **相对编码**: 使用相对位置而非绝对位置 (平移不变性)
- **动态生成**: 每次前向传播前通过 transformer 生成

#### 网格拓扑 (Mesh Topology) - `graph.face`

```python
# 输入数据中的 cells
cells: [num_triangles, 3]  # 每个三角形由 3 个节点索引组成
```

**关键特点**:
- **三角网格**: 使用三角形离散化流场区域
- **稀疏连接**: 每个节点只与邻近节点相连
- **非均匀网格**: 关键区域 (如圆柱表面) 网格更密

#### 输入数据的关键特性总结

| 特性 | 说明 | 作用 |
|------|------|------|
| **欧拉描述** | 网格点固定，预测速度场变化 | 区别于拉格朗日粒子方法 |
| **混合特征** | 物理量 (速度) + 几何类型 (节点类型) | 同时编码状态和边界条件 |
| **稀疏图结构** | 仅相邻节点通过边连接 | 局部相互作用，计算高效 |
| **归一化输入** | 在线累积统计并归一化 | 训练稳定，加速收敛 |

---

### 6.2 Rollout 详解

#### 核心概念

**Rollout = 长时序自回归推理**

```
训练时：单步预测
┌─────┐      ┌─────┐      ┌─────┐
│ v₀  │  →   │ v₁  │  →   │ v₂  │
└─────┘      └─────┘      └─────┘
   ↓            ↓            ↓
预测 v₁       预测 v₂      预测 v₃
(有真值监督)   (有真值监督)   (有真值监督)


推理时 (Rollout): 多步自回归
┌─────┐      ┌──────┐      ┌──────┐
│ v₀  │  →   │ v̂₁  │  →   │ v̂₂  │
└─────┘      └──────┘      └──────┘
   ↓            ↓            ↓
预测 v₁      预测 v̂₂      预测 v̂₃
            (用 v̂₁ 输入)   (用 v̂₂ 输入)
            ⚠️误差累积     ⚠️误差累积
```

#### Rollout 代码解读

```python
@torch.no_grad()
def rollout(model, dataset, rollout_index=1):
    num_sampes_per_tra = dataset.num_sampes_per_tra  # 轨迹长度 (~1000 步)
    predicted_velocity = None
    mask = None
    predicteds = []
    targets = []

    for i in range(num_sampes_per_tra):
        # 1. 获取当前帧的图数据
        index = rollout_index * num_sampes_per_tra + i
        graph = dataset[index]
        graph = transformer(graph)
        graph = graph.cuda()

        # 2. 获取边界条件掩码 (固定边界/入口速度)
        if mask is None:
            node_type = graph.x[:, 0]
            mask = torch.logical_or(node_type==NodeType.NORMAL, 
                                    node_type==NodeType.OUTFLOW)
            mask = torch.logical_not(mask)  # 需要固定的节点

        # 3. 【关键】自回归：用上一帧的预测作为当前帧的输入
        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()
        
        # 4. 保存真值 (用于评估误差)
        next_v = graph.y
        
        # 5. 模型预测
        with torch.no_grad():
            predicted_velocity = model(graph, velocity_sequence_noise=None)

        # 6. 【关键】边界条件约束：强制固定边界/入口的速度
        predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())
    
    return [np.stack(predicteds), np.stack(targets)]
```

#### Rollout 的挑战

```
误差累积问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
步数    预测速度    误差来源
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t=0     v₀          初始条件 (精确)
t=1     v̂₁ = f(v₀)    模型误差 ε₁
t=2     v̂₂ = f(v̂₁)   模型误差 + 误差传播 ε₁→₂
t=3     v̂₃ = f(v̂₂)   模型误差 + 累积误差 ε₁→₂→₃
...
t=1000  v̂₁₀₀₀        误差可能指数增长 ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么需要 Rollout 评估？
- 单步预测准确 ≠ 长期稳定
- Rollout 能暴露数值不稳定性和误差累积
- 物理模拟器的核心指标：能否长期稳定运行
```

### 6.3 Rollout 之间的关联性

**答案：完全独立，不会关联**

```
数据集中的样本组织:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train.npz / test.npz 结构:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

indices:   [0]────[N₁]────[N₂]────[N₃]──── ...
            │ 轨迹 0 │ 轨迹 1 │ 轨迹 2 │
            │(1000 步)│(1000 步)│(1000 步)│

rollout_index = 0: 处理轨迹 0 (样本 0~999)
rollout_index = 1: 处理轨迹 1 (样本 1000~1999)
rollout_index = 2: 处理轨迹 2 (样本 2000~2999)

每个 rollout 内部:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rollout_index=0:
  ┌─────┐   ┌─────┐   ┌─────┐        ┌──────┐
  │v₀⁰ │ → │v̂₁⁰│ → │v̂₂⁰│ → ... → │v̂₉₉₉⁰│
  └─────┘   └─────┘   └─────┘        └──────┘
  初始条件   自回归推进                    

rollout_index=1:
  ┌─────┐   ┌─────┐   ┌─────┐        ┌──────┐
  │v₀¹│ → │v̂₁¹│ → │v̂₂¹│ → ... → │v̂₉₉₉¹│
  └─────┘   └─────┘   └─────┘        └──────┘
  新的初始条件 (独立！)
```

**关键点**:

| 方面 | 说明 |
|------|------|
| **数据来源** | 每个 rollout 处理不同的轨迹样本 |
| **初始条件** | 每个轨迹有独立的初始速度场 v₀ |
| **模型参数** | 共享同一组模型权重 (但不互相影响) |
| **计算过程** | 完全独立的自回归过程 |
| **误差累积** | 每个 rollout 内部的误差不会传播到其他 rollout |

### `rollout_num` 参数的含义

```bash
python rollout.py --rollout_num 5
```

这只是指"对前 5 个测试轨迹分别进行 rollout 评估"，它们之间**没有任何数据或状态的共享**。

---

## 7. 总结

### 6.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      MeshGraphNet                           │
├─────────────────────────────────────────────────────────────┤
│  Input: mesh_pos, node_type, velocity, cells               │
│                     ↓                                       │
│  ┌─────────────┐                                           │
│  │  Encoder    │ → 特征映射到 128 维隐空间                   │
│  └─────────────┘                                           │
│         ↓                                                   │
│  ┌──────────────────────────────┐                          │
│  │  GnBlock × 15 (消息传递)      │ → 核心推理层              │
│  │  - EdgeBlock (边更新)         │                          │
│  │  - NodeBlock (节点更新)       │                          │
│  │  - 残差连接                   │                          │
│  └──────────────────────────────┘                          │
│         ↓                                                   │
│  ┌─────────────┐                                           │
│  │  Decoder    │ → 输出 2D 加速度                           │
│  └─────────────┘                                           │
│         ↓                                                   │
│  Output: predicted_velocity = velocity + acceleration      │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 训练技巧

| 技巧 | 作用 |
|------|------|
| 噪声注入 | 提高鲁棒性，模拟数值误差 |
| 在线归一化 | 无需预计算统计，数值稳定 |
| 残差连接 | 深层网络训练稳定 |
| 选择性 Loss | 仅在流体区域计算误差 |

### 7.3 可扩展性

1. **新几何**: 替换网格和节点类型
2. **新物理**: 修改输入/输出维度 (如 3D 流场、温度场)
3. **多 GPU**: 内置 DDP 支持

### 7.4 值得学习的代码设计

1. **memmap 高效数据加载** - 处理大规模数据集
2. **PyG Data 对象** - 清晰的图数据表示
3. **模块化构建块** - EdgeBlock/NodeBlock 可复用
4. **训练/推理统一接口** - `forward()` 内部分支

---

## 8. 模型结构深度解析

### 8.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MeshGraphNet 完整数据流                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  原始输入                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │ mesh_pos     │  │ node_type    │  │ velocity     │                      │
│  │ [N, 2]       │  │ [N, 1]       │  │ [N, 2]       │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│         │                │                  │                                │
│         └────────────────┼──────────────────┘                                │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    FpcDataset.__getitem__                             │ │
│  │  - 加载静态网格数据 (pos, node_type, cells)                           │ │
│  │  - 加载动态速度数据 (velocity_t, velocity_t+1)                        │ │
│  │  - 构造 PyG Data: x=[node_type+velocity], face=cells.T                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Transformer (T.Compose)                            │ │
│  │                                                                       │ │
│  │  FaceToEdge:   face[3,F] → edge_index[2,E]                           │ │
│  │  Cartesian:    添加 (dx, dy) 作为边特征                                 │ │
│  │  Distance:     添加 ||dx|| 作为边特征                                   │ │
│  │                                                                       │ │
│  │  输出：edge_attr [E, 3] = (dx, dy, distance)                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Simulator.forward()                                │ │
│  │                                                                       │ │
│  │  训练模式：                                                           │ │
│  │  1. 注入噪声：noised_vel = velocity + noise                          │ │
│  │  2. 构造节点特征：[noised_vel, onehot(node_type)] → [N, 11]          │ │
│  │  3. 归一化边特征：edge_attr → normalized                             │ │
│  │  4. GNN 前向传播：predicted_acc_norm                                 │ │
│  │  5. 计算目标：target_acc = next_vel - noised_vel                     │ │
│  │  6. 归一化目标：target_acc_norm                                      │ │
│  │  7. 返回：(predicted_acc_norm, target_acc_norm)                      │ │
│  │                                                                       │ │
│  │  推理模式：                                                           │ │
│  │  1. 构造节点特征：[clean_vel, onehot(node_type)] → [N, 11]           │ │
│  │  2. 归一化边特征                                                      │ │
│  │  3. GNN 前向传播：predicted_acc_norm                                 │ │
│  │  4. 反归一化：predicted_acc = inverse(predicted_acc_norm)            │ │
│  │  5. 速度积分：next_vel = vel + predicted_acc                         │ │
│  │  6. 返回：next_vel                                                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    EncoderProcesserDecoder                            │ │
│  │                                                                       │ │
│  │  Encoder:                                                             │ │
│  │    node_attr [N,11] → MLP → [N,128]                                  │ │
│  │    edge_attr [E,3]  → MLP → [E,128]                                  │ │
│  │                                                                       │ │
│  │  Processor (×15):                                                     │ │
│  │    for each GnBlock:                                                 │ │
│  │      EdgeBlock:  [sender, receiver, edge] → MLP → new_edge          │ │
│  │      NodeBlock:  [node, agg_edges] → MLP → new_node                 │ │
│  │      残差：new = old + transformed                                  │ │
│  │                                                                       │ │
│  │  Decoder:                                                             │ │
│  │    node_attr [N,128] → MLP → [N,2] (加速度)                          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 输入输出维度详细对照表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        各阶段张量维度变化                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  假设：N=1000 节点，E=6000 边，T=1000 时间步，B=20 batch_size               │
│                                                                             │
│  ┌────────────────────┬────────────────┬─────────────────────────────┐     │
│  │ 数据/操作          │ 输入维度       │ 输出维度                    │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ FpcDataset[0]      │ -              │ Data(x=[N,3], pos=[N,2],    │     │
│  │                    │                │ face=[3,F], y=[N,2])        │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ FaceToEdge         │ face=[3,F]     │ edge_index=[2,E]            │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ Cartesian          │ pos=[N,2],     │ edge_attr=[E,2]             │     │
│  │                    │ edge_index     │ (dx, dy)                    │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ Distance           │ edge_attr[2]   │ edge_attr=[E,3]             │     │
│  │                    │                │ (dx, dy, dist)              │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ update_node_attr   │ vel=[N,2],     │ node_attr=[N,11]            │     │
│  │                    │ type=[N,1]     │                             │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ Encoder            │ node=[N,11],   │ node=[N,128],               │     │
│  │                    │ edge=[E,3]     │ edge=[E,128]                │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ GnBlock (×15)      │ node=[N,128],  │ node=[N,128],               │     │
│  │                    │ edge=[E,128]   │ edge=[E,128]                │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ Decoder            │ node=[N,128]   │ acc=[N,2]                   │     │
│  ├────────────────────┼────────────────┼─────────────────────────────┤     │
│  │ output_normalizer  │ acc=[N,2]      │ acc_norm=[N,2]              │     │
│  │ .inverse()         │ acc_norm=[N,2] │ acc=[N,2]                   │     │
│  └────────────────────┴────────────────┴─────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 消息传递机制详解

```
Graph Network 消息传递公式 (对应 GnBlock):

┌─────────────────────────────────────────────────────────────────┐
│                    单次消息传递 (1 个 GnBlock)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Edge Update (EdgeBlock):                                      │
│  ─────────────────────                                         │
│  e'ᵢⱼ = MLPₑ(hᵢ, hⱼ, eᵢⱼ)                                      │
│                                                                 │
│  其中:                                                         │
│    hᵢ  = sender 节点特征                                       │
│    hⱼ  = receiver 节点特征                                     │
│    eᵢⱼ = 边 (i→j) 的特征                                        │
│    e'ᵢⱼ = 更新后的边特征                                        │
│                                                                 │
│  Node Update (NodeBlock):                                      │
│  ───────────────────                                           │
│  h'ⱼ = MLPₙ(hⱼ, Σᵢ∈N(j) e'ᵢⱼ)                                │
│                                                                 │
│  其中:                                                         │
│    hⱼ  = 节点 j 的原始特征                                      │
│    N(j) = 所有以 j 为接收者的发送者节点集合                     │
│    Σ = scatter_add (累加所有入边特征)                          │
│    h'ⱼ = 更新后的节点特征                                       │
│                                                                 │
│  Residual Connection:                                          │
│  ───────────────────                                           │
│  hⱼ = hⱼ + h'ⱼ                                                 │
│  eᵢⱼ = eᵢⱼ + e'ᵢⱼ                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

15 层消息传递的感受野:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
层数    聚合范围          说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1       1 跳邻域         直接相邻节点
2       2 跳邻域         邻居的邻居
3       3 跳邻域         ...
...
15      ~15 跳邻域       几乎覆盖整个计算域
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对于典型圆柱绕流网格 (约 2000 节点):
- 15 跳足以让圆柱表面的信息传播到尾流区域
- 这是选择 15 层的经验依据
```

### 8.4 边界条件处理机制

```
边界条件实现方式:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 训练时的隐式学习:
   - 边界节点类型 (INFLOW, WALL 等) 作为 one-hot 输入
   - 模型学习"看到"这些类型时应输出的速度模式

2. 推理时的显式约束 (rollout.py):
   ```python
   # 识别需要固定的节点 (非 NORMAL 且非 OUTFLOW)
   mask = (node_type != NORMAL) & (node_type != OUTFLOW)
   
   # 强制重置为边界条件值
   predicted_velocity[mask] = boundary_value[mask]
   ```

边界类型与典型速度约束:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
类型              速度约束              物理意义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFLOW (4)       v = v_inlet (给定)    入口来流条件
WALL_BOUNDARY (6) v = 0 (无滑移)      固壁边界
OBSTACLE (1)     v = 0 (无滑移)      障碍物表面
AIRFOIL (2)      v = 0 (无滑移)      翼型表面
OUTFLOW (5)      ∂v/∂n = 0 (自由出流) 出口，模型预测
NORMAL (0)       -                    流体区域，模型预测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. 示例 Case 集合

### Case 1: 单步推理 (未训练模型)

```python
"""
场景：快速测试模型接口是否正常工作
目的：验证数据流通过程，不关心预测质量
"""
import torch
from model.simulator import Simulator
from torch_geometric.data import Data

# 1. 创建随机模型
model = Simulator(
    message_passing_num=15,
    node_input_size=11,
    edge_input_size=3,
    device='cpu'
)
model.eval()

# 2. 创建测试数据
N, E = 100, 300  # 100 节点，300 边
graph = Data(
    x=torch.randn(N, 11),           # 随机节点特征
    pos=torch.randn(N, 2),          # 随机位置
    edge_index=torch.randint(0, N, (2, E)),  # 随机边
    edge_attr=torch.randn(E, 3),    # 随机边特征
    y=torch.randn(N, 2)             # 随机目标
)

# 3. 推理
with torch.no_grad():
    output = model(graph, None)

print(f"输入：{graph.x.shape} → 输出：{output.shape}")
# 期望输出：torch.Size([100, 11]) → torch.Size([100, 2])
```

### Case 2: 完整训练 + 推理流程

```python
"""
场景：端到端测试整个训练和推理流程
目的：验证从数据加载到 rollout 的完整链路
"""
from dataset import FpcDataset
from model.simulator import Simulator
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch

# 配置
data_root = "data"
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 加载数据
dataset = FpcDataset(data_root, split='train')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. 创建模型
model = Simulator(
    message_passing_num=15,
    node_input_size=11,
    edge_input_size=3,
    device=device
).to(device)

# 3. 预处理变换
transformer = T.Compose([
    T.FaceToEdge(),
    T.Cartesian(norm=False),
    T.Distance(norm=False)
])

# 4. 单步训练演示
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

for batch in loader:
    batch = transformer(batch).to(device)
    
    # 前向传播
    pred, target = model(batch, velocity_sequence_noise=0.02)
    
    # 计算 loss
    loss = torch.mean((pred - target) ** 2)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Step loss: {loss.item():.6f}")
    break  # 仅演示一步

# 5. 推理演示
model.eval()
with torch.no_grad():
    sample = dataset[0]
    sample = transformer(sample).unsqueeze(0).to(device)
    prediction = model(sample, None)
    print(f"Prediction shape: {prediction.shape}")
```

### Case 3: 多步 Rollout 误差分析

```python
"""
场景：评估模型长期预测稳定性
目的：观察误差随时间的累积情况
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_rollout_error(predicteds, targets):
    """
    分析 rollout 误差累积曲线
    
    Args:
        predicteds: [T, N, 2] 预测序列
        targets: [T, N, 2] 目标序列
    """
    T = len(predicteds)
    errors = []
    
    for t in range(T):
        # 计算当前步的 RMSE
        mse = np.mean((predicteds[t] - targets[t]) ** 2)
        rmse = np.sqrt(mse)
        errors.append(rmse)
    
    # 绘制误差曲线
    plt.figure(figsize=(10, 4))
    plt.plot(errors, 'b-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('Rollout Error Accumulation')
    plt.grid(True, alpha=0.3)
    plt.savefig('rollout_error.png')
    
    # 打印关键点
    for t in [0, T//4, T//2, 3*T//4, T-1]:
        print(f"Step {t}: RMSE = {errors[t]:.6f}")
    
    return errors

# 使用示例 (假设已有 rollout 结果)
# errors = analyze_rollout_error(predicteds, targets)
```

### Case 4: 节点类型可视化

```python
"""
场景：理解网格中不同节点类型的分布
目的：直观查看边界条件位置
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def visualize_node_types(pos, node_type, cells=None):
    """
    可视化节点类型分布
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 颜色映射
    type_names = ['NORMAL', 'OBSTACLE', 'AIRFOIL', 'HANDLE', 
                  'INFLOW', 'OUTFLOW', 'WALL']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#7f7f7f']
    
    # 绘制网格
    if cells is not None:
        triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=cells)
        ax.triplot(triang, 'k-', alpha=0.3, lw=0.5)
    
    # 按类型绘制节点
    for nt, name in enumerate(type_names):
        mask = (node_type.flatten() == nt)
        if np.any(mask):
            ax.scatter(pos[mask, 0], pos[mask, 1], 
                      c=[colors[nt]], label=name, s=50,
                      edgecolors='black', linewidth=0.5)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Node Type Distribution')
    plt.tight_layout()
    plt.savefig('node_types.png', dpi=150)
    plt.show()

# 使用示例
# visualize_node_types(pos, node_type, cells)
```

---

## 11. ONNX 导出

### 11.1 为什么需要 ONNX 导出？

| 用途 | 说明 |
|------|------|
| **模型可视化** | 使用 Netron 直观查看模型架构 |
| **跨平台部署** | 在 C++、JavaScript、移动端部署 |
| **推理加速** | 使用 ONNX Runtime 获得更高性能 |
| **调试分析** | 检查每一层的输出和中间张量 |

### 11.2 导出挑战与解决方案

MeshGraphNet 基于 PyTorch Geometric，导出 ONNX 面临以下挑战：

| 挑战 | 解决方案 |
|------|----------|
| **torch_scatter.scatter_add 不可导出** | 实现纯 PyTorch 版本 `scatter_add_onnx()` |
| **PyG Data 对象不兼容** | 创建使用张量接口的 `*ONNX` 类 |
| **动态图结构** | 显式传递 `num_nodes` 参数 |

### 11.3 使用方法

```bash
# 导出随机权重模型（测试架构）
python export_onnx.py --output model.onnx

# 导出训练好的模型
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx --visualize

# 可视化（使用 Netron）
pip install netron
netron model.onnx
# 或在浏览器打开 https://netron.app/ 并上传 model.onnx
```

### 11.4 ONNX 模型规格

**输入输出**:

| 名称 | 形状 | 说明 |
|------|------|------|
| `node_attr` | `[N, 11]` | 节点特征（速度 2 + 节点类型 one-hot 9） |
| `edge_attr` | `[E, 3]` | 边特征（dx, dy, distance） |
| `edge_index` | `[2, E]` | 边索引 |
| `acceleration` | `[N, 2]` | 加速度预测 |

**模型统计**:
- 总算子数：~1,200
- Gemm (全连接): 132 层
- Relu (激活): 99 层
- ScatterElements (消息聚合): 15 层

### 11.5 测试 ONNX 推理

```bash
# 运行测试脚本
python test_onnx.py

# 预期输出:
# ✓ 模型加载成功
# ✓ 推理成功!
# 平均推理时间：~7.8ms
```

### 11.6 相关文档

- [`docs/onnx-export-guide.md`](./onnx-export-guide.md) - ONNX 导出详细指南
- [`docs/onnx-export-summary.md`](./onnx-export-summary.md) - 实现总结

---

## 12. 新增文档说明

### 12.1 论文解读

- **文件**: [`docs/meshgraphnets-paper-explained.md`](./meshgraphnets-paper-explained.md)
- **内容**: 面向入门者的论文解读
  - 核心思想：网格→图→GNN 消息传递
  - 模型架构详解（Encoder-Processor-Decoder）
  - 关键设计：为什么预测加速度
  - 训练技巧、实验结果
  - 入门者学习路径和术语表

### 12.2 数据结构详解

- **文件**: [`docs/data-structure-explained.md`](./data-structure-explained.md)
- **内容**: 5 种数据类型的详细解释和模型消费流程

### 12.3 数据集分析

- **文件**: [`docs/dataset-analysis.md`](./dataset-analysis.md)
- **内容**: train/valid/test 三数据集的对比分析
  - 关键发现：测试集速度范围是训练集的 4 倍（需要外推能力）

---

## 13. 常见问题与调试技巧

### Q1: 输出全是 NaN 怎么办？

**可能原因**:
1. 学习率过大 → 尝试降低到 1e-4 或 1e-5
2. 归一化统计量异常 → 检查 `_acc_count` 是否正常累积
3. 网格质量问题 → 检查是否有重合节点或退化单元

**调试代码**:
```python
# 检查归一化器状态
print(f"Acc count: {model._output_normalizer._acc_count}")
print(f"Acc sum: {model._output_normalizer._acc_sum}")
print(f"Mean: {model._output_normalizer._mean()}")
print(f"Std: {model._output_normalizer._std_with_epsilon()}")
```

### Q2: Loss 不下降怎么办？

**检查清单**:
1. ✅ 确认节点类型 one-hot 编码正确
2. ✅ 确认边特征已生成 (检查 `graph.edge_attr` 维度)
3. ✅ 确认 mask 计算正确 (检查 `NodeType` 枚举值)
4. ✅ 确认噪声注入只在训练时启用

### Q3: 如何调试消息传递过程？

**添加 Hook 观察中间输出**:
```python
def register_debug_hooks(model):
    def make_hook(name):
        def hook(module, input, output):
            print(f"{name}:")
            print(f"  input[0].shape: {input[0].shape}")
            print(f"  output.shape: {output.shape}")
            print(f"  output mean: {output.mean().item():.6f}")
            print(f"  output std: {output.std().item():.6f}")
        return hook
    
    for i, block in enumerate(model.model.processer_list):
        block.nb_module.register_forward_hook(
            make_hook(f'GnBlock{i}-NodeBlock')
        )

# 使用
register_debug_hooks(model)
```

---

## 14. 参考资源

### 核心资源

- **原论文**: https://arxiv.org/abs/2010.03409
- **DeepMind 数据**: https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **图神经网络教程**: https://davidstutz.de/lectures-on-graph-neural-networks/

### 本项目文档

| 文档 | 说明 |
|------|------|
| [`understanding-notes.md`](./understanding-notes.md) | 详细代码理解笔记（本文档） |
| [`data-structure-explained.md`](./data-structure-explained.md) | 数据结构与模型消费流程详解 |
| [`dataset-analysis.md`](./dataset-analysis.md) | 数据集详细分析（train/valid/test 对比） |
| [`onnx-export-guide.md`](./onnx-export-guide.md) | ONNX 导出完整指南 |
| [`meshgraphnets-paper-explained.md`](./meshgraphnets-paper-explained.md) | 论文入门解读 |

### 工具

- **Netron**: https://netron.app/ - ONNX 模型可视化
- **ONNX Runtime**: https://onnxruntime.ai/ - 跨平台推理加速
