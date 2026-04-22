# Rollout 推理数据流详解

**日期**: 2026-04-22  
**目的**: 深入理解 rollout 推理的每一步数据变化，明确输入、输出和保存内容

---

## 核心概念

### 什么是 Rollout？

Rollout 是指从初始条件开始，反复使用模型预测下一时刻状态，并将**预测结果作为下一步的输入**，如此循环进行长期预测。

```
普通推理 vs Rollout 推理:

普通推理：输入真值 → 模型 → 输出预测    (单步，独立)
Rollout:   输入预测 → 模型 → 输出预测    (多步，自回归)
           ↑___________________|
           上一步的预测作为输入
```

### 为什么需要 Rollout 评估？

| 原因 | 说明 |
|------|------|
| 模拟真实使用场景 | 实际使用时就是自回归推理，没有真值可用 |
| 检测误差累积 | 单步准确 ≠ 长期稳定，误差会随时间累积 |
| 评估数值稳定性 | 检查误差是否随时间指数增长导致发散 |

---

## 数据结构定义

### 数据集中的样本结构

```python
# dataset[index] 返回一个 PyG Data 对象:

graph.x[:, 0:1]   # 节点类型 [N, 1]
                  #   0=NORMAL, 1=OBSTACLE, 2=AIRFOIL,
                  #   3=HANDLE, 4=INFLOW, 5=OUTFLOW, 6=WALL_BOUNDARY

graph.x[:, 1:3]   # 【当前帧速度】[N, 2] - 记作 v(t)
                  # 这是**输入特征**

graph.y           # 【下一帧速度】[N, 2] - 记作 v(t+1)
                  # 这是**预测目标/真值**

graph.pos         # 节点位置 [N, 2]
graph.face        # 三角网格 [3, F]
```

### 关键区别

```
┌─────────────────────────────────────────────────────────┐
│  graph.x[:, 1:3] ≠ graph.y                              │
│                                                         │
│  graph.x[:, 1:3] = 当前时刻 t 的速度 (输入)             │
│  graph.y         = 下一时刻 t+1 的速度 (目标/真值)      │
└─────────────────────────────────────────────────────────┘
```

---

## Rollout 流程详解

### 整体流程

```python
predicted_velocity = None  # 上一帧的预测速度
mask = None                # 边界条件掩码
predicteds = []            # 预测序列
targets = []               # 目标序列

for i in range(num_steps):  # 599 步
    # 1. 加载当前帧数据
    graph = dataset[index]
    
    # 2. 获取边界条件掩码 (仅第一步)
    if mask is None:
        node_type = graph.x[:, 0]
        mask = torch.logical_not(
            torch.logical_or(
                node_type == NodeType.NORMAL,
                node_type == NodeType.OUTFLOW
            )
        )
    
    # 3. 【关键】自回归输入
    if predicted_velocity is not None:
        graph.x[:, 1:3] = predicted_velocity.detach()
    
    # 4. 获取真值 (用于边界约束)
    next_v = graph.y
    
    # 5. 模型预测
    with torch.no_grad():
        predicted_velocity = model(graph)
    
    # 6. 【关键】边界条件约束
    predicted_velocity[mask] = next_v[mask]
    
    # 7. 保存结果
    predicteds.append(predicted_velocity.cpu().numpy())
    targets.append(next_v.cpu().numpy())

# 最终保存
result = [np.stack(predicteds), np.stack(targets)]
```

---

## Step-by-Step 数据流

### Step 0 (第一步)

```
┌─────────────────────────────────────────────────────────┐
│ 输入：v(0) - 初始条件 (已知)                            │
│ 输出：v_pred(1) - 预测的第 1 步速度                      │
│ 保存：v_pred(1), v(1)                                   │
└─────────────────────────────────────────────────────────┘
```

```python
# 1. 加载数据
graph = dataset[0]
#   graph.x[:, 1:3] = v(0)  ← 初始速度场
#   graph.y         = v(1)  ← 第 1 步真值

# 2. 自回归输入检查
if predicted_velocity is not None:  # False，第一步
    pass  # 不执行，保持 graph.x[:, 1:3] = v(0)

# 3. 模型输入
input = graph.x[:, 1:3]  # = v(0)

# 4. 模型预测
predicted_velocity = model(graph)  # 输出：v_pred(1)

# 5. 边界约束
predicted_velocity[mask] = graph.y[mask]
# 边界节点用真值 v(1) 替换

# 6. 保存
predicteds.append(v_pred(1))  # 保存预测的第 1 步
targets.append(v(1))          # 保存真值第 1 步
```

---

### Step 1 (第二步)

```
┌─────────────────────────────────────────────────────────┐
│ 输入：v_pred(1) - 上一步的预测值 (不是真值!)            │
│ 输出：v_pred(2) - 预测的第 2 步速度                      │
│ 保存：v_pred(2), v(2)                                   │
└─────────────────────────────────────────────────────────┘
```

```python
# 1. 加载数据
graph = dataset[1]
#   graph.x[:, 1:3] = v(1)  ← 第 1 步真值速度
#   graph.y         = v(2)  ← 第 2 步真值

# 2. 【关键】自回归输入
if predicted_velocity is not None:  # True
    graph.x[:, 1:3] = predicted_velocity.detach()
    # 替换为：graph.x[:, 1:3] = v_pred(1)  ← 用预测值!

# 3. 模型输入
input = graph.x[:, 1:3]  # = v_pred(1)  ← 不是真值 v(1)!

# 4. 模型预测
predicted_velocity = model(graph)  # 输出：v_pred(2)

# 5. 边界约束
predicted_velocity[mask] = graph.y[mask]
# 边界节点用真值 v(2) 替换

# 6. 保存
predicteds.append(v_pred(2))
targets.append(v(2))
```

---

### Step 2 及以后

```
Step 2:
  输入：v_pred(2) ← 来自 Step 1 的预测
  输出：v_pred(3)
  保存：v_pred(3), v(3)

Step 3:
  输入：v_pred(3) ← 来自 Step 2 的预测
  输出：v_pred(4)
  保存：v_pred(4), v(4)

... 以此类推直到 Step 598
```

---

## 数据流可视化

```
时间线：t=0         t=1         t=2         t=3
        │          │          │          │
        ↓          │          │          │
      v(0)【真】   │          │          │   ← 初始条件
        │          │          │          │
        ├─→ Model ─┤          │          │
        │          ↓          │          │
        │     v_pred(1)       │          │   ← 模型预测
        │          │          │          │
        │          ├─→ 边界约束           │
        │          │   v_pred(1)[mask] = v(1)[mask]
        │          │          │          │
        │          ├─→ 保存              │
        │          │   predicteds[0] = v_pred(1)
        │          │   targets[0]    = v(1)
        │          │          │          │
        │          └──┐    ┌─┘          │
        │             │    │            │
        │             └────┼────────────┤
        │                  │            │
        │                  ↓            │
        │             v_pred(1)【预测】 │   ← 自回归关键点!
        │                  │            │
        │                  ├─→ Model ──┤
        │                  │           ↓
        │                  │       v_pred(2)
        │                  │            │
        │                  ├─→ 边界约束
        │                  │   v_pred(2)[mask] = v(2)[mask]
        │                  │            │
        │                  ├─→ 保存
        │                  │   predicteds[1] = v_pred(2)
        │                  │   targets[1]    = v(2)
        │                  │            │
        │                  └──┐    ┌──┘
        │                     │    │
        │                     └────┼───────────┤
        │                          │           │
        │                          ↓           │
        │                     v_pred(2)【预测】← 继续自回归
        │                          │
        │                          └─→ ...
        │
```

---

## 最终保存的内容

### PKL 文件结构

```python
# 保存到 result/result{index}.pkl
[result, crds] = pickle.load(f)

# result = [predicteds, targets]
result = [
    np.stack([v_pred(1), v_pred(2), ..., v_pred(599)]),  # predicteds
    np.stack([v(1), v(2), ..., v(599)])                   # targets
]

# crds = 网格坐标
crds = graph.pos.cpu().numpy()  # [N, 2]
```

### 数据形状

```python
# predicteds: 预测序列
predicteds.shape  # [599, N, 2]
                  # 599 步 × N 个节点 × 2D 速度分量

# targets: 真值序列
targets.shape     # [599, N, 2]
                  # 599 步 × N 个节点 × 2D 速度分量

# coordinates: 网格坐标
crds.shape        # [N, 2]
                  # N 个节点的 2D 位置
```

### 为什么是 599 步而不是 600 步？

```
数据集有 600 个样本 (index 0-599):
  dataset[0]   → v(0) → v(1)
  dataset[1]   → v(1) → v(2)
  ...
  dataset[599] → v(599) → v(600)

Rollout 从初始条件 v(0) 开始预测:
  Step 0: 输入 v(0) → 预测 v(1)
  Step 1: 输入 v_pred(1) → 预测 v(2)
  ...
  Step 598: 输入 v_pred(598) → 预测 v(599)

所以保存的是 [v_pred(1), ..., v_pred(599)] 共 599 个预测结果
```

---

## 关键设计决策

### 1. 为什么输入用预测值而不是真值？

```
┌─────────────────────────────────────────────────────────┐
│ 自回归 (Autoregressive) 的核心思想                       │
│                                                         │
│ 训练时：可以用真值作为输入 (Teacher Forcing)            │
│ 推理时：没有真值可用，只能用预测值                      │
│                                                         │
│ Rollout 模拟的是真实使用场景，所以必须用预测值!         │
└─────────────────────────────────────────────────────────┘
```

### 2. 为什么边界条件要用真值重置？

```
物理边界条件必须满足：

入口 (INFLOW): 流速固定为给定值
壁面 (WALL): 速度为 0 (无滑移条件)
障碍物 (OBSTACLE): 速度为 0

如果不用真值重置，模型预测的边界值会漂移，
导致物理上不允许的结果。
```

### 3. 为什么要同时保存 predicteds 和 targets？

| 保存内容 | 用途 |
|----------|------|
| predicteds | 模型的预测序列，用于可视化、分析 |
| targets | 真值序列，用于计算误差、评估性能 |

---

## 误差累积分析

### 典型误差增长曲线

```
Step    Max Diff    说明
─────   ─────────   ─────────────────────────────────
0       ~3e-08      第一步，误差极小
1       ~6e-08      第二步，开始累积
10      ~3e-07      误差线性增长
50      ~1.5e-06    继续增长
100     ~3e-06
200     ~6e-06
400     ~1.2e-05
598     ~1.8e-05    最终误差 (仍在可接受范围)
```

### 误差来源

1. **模型预测误差**: 模型本身的不完美
2. **浮点数舍入误差**: 不同计算顺序导致微小差异
3. **累积效应**: 每一步的误差会传递到下一步

---

## 验证结果

### V1/V2/V3/V4 内部一致性

| 对比 | max_diff | 结果 |
|------|----------|------|
| V1 vs V2 | 3.57e-05 | ✅ |
| V1 vs V3 | 3.57e-05 | ✅ |
| V1 vs V4 | 1.79e-05 | ✅ |

### batch rollout vs 原始 rollout

| 对比 | max_diff | 结果 |
|------|----------|------|
| rollout.py vs batch_rollout.py | 1.79e-05 | ✅ |

**结论**: 所有实现方式输出一致，多 batch 并行推理正确！

---

## 相关代码文件

| 文件 | 说明 |
|------|------|
| `rollout.py` | 原始单轨迹 rollout 实现 |
| `batch_rollout.py` (V1) | 手动索引的 batch rollout |
| `batch_rollout_v2.py` | Subset + DataLoader 实现 |
| `batch_rollout_v3.py` | TrajectoryDataset 封装 |
| `batch_rollout_v4.py` | 3D 张量预加载实现 |
| `compare_rollout_results.py` | 结果比较工具 |
| `verify_rollouts.py` | 自动验证脚本 |

---

## 总结

Rollout 推理的核心是**自回归**：

```
输入：上一步的预测值 (不是真值!)
输出：当前步的预测值
保存：预测值 + 真值 (用于评估)
边界：强制重置为真值 (物理约束)
```

这种设计模拟了真实使用场景，能够评估模型的长期预测稳定性和误差累积行为。
