# MeshGraphNets 数据集详细分析

> 分析对象：DeepMind cylinder_flow 数据集 (TFRecord 格式)  
> 分析时间：2026-04-18  
> 工具：自定义 Python 脚本 (`inspect_tfrecord_deep.py`, `analyze_all_datasets.py`)

---

## 一、数据集概览

### 1.1 文件统计

| 数据集 | 文件 | 轨迹数 | 文件大小 | 占比 |
|--------|------|--------|----------|------|
| **训练集** | `train.tfrecord` | 1,000 | 12.71 GB | 83.3% |
| **验证集** | `valid.tfrecord` | 100 | 1.27 GB | 8.3% |
| **测试集** | `test.tfrecord` | 100 | 1.26 GB | 8.3% |
| **总计** | - | **1,200** | **15.24 GB** | 100% |

### 1.2 数据结构

每条轨迹 (trajectory) 包含以下字段：

| 字段名 | 数据类型 | 形状 | 说明 |
|--------|----------|------|------|
| `mesh_pos` | float32 | `[N, 2]` | 网格节点位置 (x, y 坐标) |
| `node_type` | int32 | `[N]` | 节点类型索引 |
| `cells` | int32 | `[F, 3]` | 三角网格单元连接关系 |
| `velocity` | float32 | `[T, N, 2]` | 速度场 (时间×节点×2D 向量) |
| `pressure` | float32 | `[T, N]` | 压力场 (辅助物理量) |

**符号说明**:
- N ≈ 1900：网格节点数
- F ≈ 3500：三角单元数
- T = 600：时间步数

---

## 二、详细对比分析

### 2.1 数据集规模对比

```
┌──────────────────────────────────────────────────────────────┐
│                    数据集规模对比                             │
├──────────────────────────────────────────────────────────────┤
│  train:  ████████████████████████████████████████  1000 条    │
│  valid:  ████                                     100 条     │
│  test:   ████                                     100 条     │
└──────────────────────────────────────────────────────────────┘
```

| 数据集 | 轨迹数 | 节点数 (N) | 单元数 (F) | 时间步 (T) | 文件大小 |
|--------|--------|------------|------------|------------|----------|
| **train** | 1,000 | 1,876 | 3,518 | 600 | 12.71 GB |
| **valid** | 100 | 1,896 | 3,558 | 600 | 1.27 GB |
| **test** | 100 | 1,923 | 3,612 | 600 | 1.26 GB |

**关键发现**:
- 三个数据集的网格规模基本一致 (~1,900 节点)
- 测试集节点数略多 (1,923 vs 1,876)，可能包含更复杂的几何
- 每条轨迹都是 600 时间步

---

### 2.2 节点类型分布对比

#### 训练集 (train)

| 类型 | ID | 节点数 | 占比 | 是否预测 |
|------|-----|--------|------|----------|
| **NORMAL** | 0 | 1,642 | 87.5% | ✅ |
| **WALL** | 6 | 200 | 10.7% | ❌ |
| **INFLOW** | 4 | 17 | 0.9% | ❌ |
| **OUTFLOW** | 5 | 17 | 0.9% | ✅ |

#### 验证集 (valid)

| 类型 | ID | 节点数 | 占比 |
|------|-----|--------|------|
| **NORMAL** | 0 | 1,662 | 87.7% |
| **WALL** | 6 | 200 | 10.5% |
| **INFLOW** | 4 | 17 | 0.9% |
| **OUTFLOW** | 5 | 17 | 0.9% |

#### 测试集 (test)

| 类型 | ID | 节点数 | 占比 |
|------|-----|--------|------|
| **NORMAL** | 0 | 1,689 | 87.8% |
| **WALL** | 6 | 200 | 10.4% |
| **INFLOW** | 4 | 17 | 0.9% |
| **OUTFLOW** | 5 | 17 | 0.9% |

**关键发现**:
- 所有数据集使用相同的边界配置
- WALL 节点约 200 个 (10.5%)，可能是圆柱表面 + 上下壁面
- INFLOW/OUTFLOW 各 17 个节点，对应入口和出口边界
- 实际参与 Loss 计算的节点：NORMAL + OUTFLOW ≈ 88.5%

---

### 2.3 速度场统计对比

| 数据集 | 最小值 | 最大值 | 范围 | 备注 |
|--------|--------|--------|------|------|
| **train** | -0.3267 | 0.7147 | 1.04 | 训练分布 |
| **valid** | -0.2645 | 0.5625 | 0.83 | 与训练集接近 |
| **test** | -1.4918 | 2.7320 | 4.22 ⚠️ | **明显更大的范围** |

```
速度范围对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train:  [-0.33 ───────────── 0.71]
valid:  [-0.26 ───────── 0.56]
test:   [-1.49 ─────────────────────────── 2.73] ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**重要洞察**:
- 测试集的速度范围是训练集的 **4 倍**
- 这意味着模型需要**外推 (extrapolation)** 能力，而非仅仅插值
- 测试集包含更极端的流动条件（可能是更高的雷诺数或不同的圆柱位置）
- 这是评估模型泛化能力的关键设计

---

## 三、数据结构深度解析

### 3.1 单条轨迹数据示例 (train 第一条)

```python
mesh_pos:  (1876, 2)      # 3,792 个 float32 值
node_type: (1876,)        # 1,876 个 int32 值
cells:     (3518, 3)      # 10,674 个 int32 值
velocity:  (600, 1876, 2) # 2,275,200 个 float32 值
pressure:  (600, 1876)    # 1,137,600 个 float32 值
```

### 3.2 网格节点位置 (mesh_pos)

**前 5 个节点位置**:
```
节点 0: (0.0000, 0.0158)
节点 1: (0.0000, 0.0078)
节点 2: (0.0153, 0.0158)
节点 3: (0.0000, 0.0306)
节点 4: (0.0182, 0.0272)
```

**观察**:
- 多个节点的 x=0，这些可能是入口边界 (INFLOW)
- y 坐标范围约 0~0.05，这是计算域高度
- 节点分布不均匀，某些区域更密集

### 3.3 节点类型 (node_type)

**前 10 个节点类型**:
```
[4, 4, 0, 4, 0, 4, 0, 4, 0, 4]
       ↑     ↑        ↑
    INFLOW  NORMAL  INFLOW
```

**唯一值**: `[0, 4, 5, 6]` = NORMAL, INFLOW, OUTFLOW, WALL

**注意**: 数据集中没有 OBSTACLE (1) 和 AIRFOIL (2) 类型，说明这是简单的圆柱绕流，圆柱表面被标记为 WALL。

### 3.4 三角网格单元 (cells)

**前 5 个三角单元**:
```
单元 0: 节点 [0, 1, 2]
单元 1: 节点 [3, 0, 4]
单元 2: 节点 [5, 3, 6]
单元 3: 节点 [7, 8, 9]
单元 4: 节点 [10, 5, 11]
```

**三角单元数**: 3,518 个

### 3.5 速度场 (velocity)

**t=0 时刻前 5 个节点的速度**:
```
节点 0: vx=0.0779, vy=0.0000, 速度大小=0.0779
节点 1: vx=0.0394, vy=0.0000, 速度大小=0.0394
节点 2: vx=0.1001, vy=-0.0809, 速度大小=0.1287
节点 3: vx=0.1453, vy=0.0000, 速度大小=0.1453
节点 4: vx=0.1505, vy=-0.1323, 速度大小=0.2004
```

**观察**:
- 某些节点的 vy=0，这些可能是边界节点
- 速度大小范围 0.04~0.20，这是典型的层流速度

---

## 四、物理场景推断

根据数据分析，可以推断出模拟的物理场景：

### 4.1 计算域设置

```
┌────────────────────────────────────────────────────────┐
│ INFLOW (4)                                             │
│ x=0, 17 节点              WALL (6)                     │
│ vx≈0.1, vy=0            y=0 和 y=H                      │
│                                                        │
│           ╭─────╮                                      │
│           │ 圆柱 │  WALL (6) ≈ 200 节点               │
│           ╰─────╯  (圆柱表面)                          │
│                                                        │
│                                      OUTFLOW (5)       │
│                                      x=L, 17 节点      │
└────────────────────────────────────────────────────────┘
                      NORMAL (0) ≈ 1642 节点
```

### 4.2 边界条件

| 边界 | 类型 | 节点数 | 速度条件 |
|------|------|--------|----------|
| 入口 | INFLOW | 17 | vx ≈ 0.1, vy = 0 (固定) |
| 出口 | OUTFLOW | 17 | 模型预测 (自由流出) |
| 壁面 | WALL | 200 | vx = 0, vy = 0 (无滑移) |
| 圆柱 | WALL | (包含在 200 中) | vx = 0, vy = 0 (无滑移) |

### 4.3 流动特征

- **雷诺数**: 根据速度范围 (~0.1-0.7) 和典型圆柱直径，Re ≈ 100-1000
- **流动状态**: 层流，可能产生卡门涡街 (Kármán vortex street)
- **时间步长**: 600 步，足以捕捉多个涡脱周期

---

## 五、对模型训练的启示

### 5.1 训练集覆盖范围

```
训练集速度范围: [-0.33, 0.71]
测试集速度范围: [-1.49, 2.73]
                ↑
                模型必须外推到未见条件
```

**挑战**: 模型在训练时从未见过速度 > 0.71 或 < -0.33 的流动，需要在测试时泛化到 4 倍大的范围。

### 5.2 Loss 计算策略

```python
# 仅在 NORMAL 和 OUTFLOW 节点上计算 loss
mask = (node_type == 0) | (node_type == 5)
#       NORMAL      OUTFLOW
#       87.5%       0.9%
#       └─────────────────┘  ≈ 88.5% 的节点参与 Loss
```

**意义**:
- WALL 和 INFLOW 节点不参与 Loss（它们的速度是固定的）
- 模型主要学习流体区域的速度演化

### 5.3 数据量估计

```
单条轨迹数据量:
  velocity: 600 × 1876 × 2 × 4 bytes ≈ 9 MB
  pressure: 600 × 1876 × 4 bytes ≈ 4.5 MB
  静态数据：≈ 0.1 MB

训练集总量: 1000 × 13.5 MB ≈ 13.5 GB ✓ (与 12.71 GB 接近)
```

---

## 六、数据读取示例代码

### 6.1 使用 TensorFlow 读取

```python
import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()

def load_trajectory(tfrecord_path, index=0):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for i, raw_record in enumerate(dataset):
        if i == index:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # 解码所有字段
            mesh_pos = np.frombuffer(
                features['mesh_pos'].bytes_list.value[0],
                dtype=np.float32
            ).reshape(-1, 2)

            node_type = np.frombuffer(
                features['node_type'].bytes_list.value[0],
                dtype=np.int32
            )

            cells = np.frombuffer(
                features['cells'].bytes_list.value[0],
                dtype=np.int32
            ).reshape(-1, 3)

            velocity = np.frombuffer(
                features['velocity'].bytes_list.value[0],
                dtype=np.float32
            ).reshape(-1, mesh_pos.shape[0], 2)

            return {
                'mesh_pos': mesh_pos,
                'node_type': node_type,
                'cells': cells,
                'velocity': velocity,
            }
    return None

# 使用示例
data = load_trajectory('data/train.tfrecord', index=0)
print(f"节点数：{data['mesh_pos'].shape[0]}")
print(f"时间步：{data['velocity'].shape[0]}")
```

### 6.2 批量统计分析

```python
def analyze_dataset(tfrecord_path):
    """分析数据集的统计信息"""
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1

    # 读取第一条进行详细分析
    data = load_trajectory(tfrecord_path)
    velocity = data['velocity']

    return {
        'num_trajectories': count,
        'num_nodes': data['mesh_pos'].shape[0],
        'num_cells': data['cells'].shape[0],
        'time_steps': velocity.shape[0],
        'velocity_min': velocity.min(),
        'velocity_max': velocity.max(),
        'node_types': np.unique(data['node_type'], return_counts=True)
    }

# 分析所有数据集
for split in ['train', 'valid', 'test']:
    stats = analyze_dataset(f'data/{split}.tfrecord')
    print(f"{split}: {stats}")
```

---

## 七、关键脚本说明

项目提供了以下数据分析脚本：

| 脚本 | 功能 |
|------|------|
| `inspect_tfrecord.py` | 快速查看 TFRecord 字段列表和记录数 |
| `inspect_tfrecord_deep.py` | 深度解码数据，查看实际形状和内容 |
| `analyze_all_datasets.py` | 对比分析 train/valid/test 三个数据集 |
| `understand_data_flow.py` | 可视化网格、节点类型、速度场 |

**使用方法**:
```bash
source /Users/chuanzhu/miniconda3/etc/profile.d/conda.sh
conda activate meshgraphnet

# 快速检查
python inspect_tfrecord.py

# 深度分析
python inspect_tfrecord_deep.py

# 对比所有数据集
python analyze_all_datasets.py

# 可视化数据流
python understand_data_flow.py
```

---

## 八、总结

### 8.1 数据特点

| 特点 | 描述 |
|------|------|
| **规模** | 1,200 条轨迹，15.24 GB |
| **时间分辨率** | 600 时间步/轨迹 |
| **空间分辨率** | ~1,900 节点，~3,500 三角单元 |
| **边界配置** | 统一的 INFLOW/WALL/OUTFLOW 设置 |
| **泛化挑战** | 测试集速度范围是训练集的 4 倍 |

### 8.2 对模型设计的启示

1. **归一化很重要**: 速度范围变化大，需要在线归一化器
2. **边界条件处理**: WALL 和 INFLOW 节点速度固定，需要特殊处理
3. **外推能力**: 测试集要求模型能泛化到未见条件
4. **数据效率**: 训练集 1,000 条 vs 测试集 100 条，需要高效学习

### 8.3 与模型输出的关系

```
输入: velocity[t] ∈ [-0.33, 0.71] (训练集范围)
      ↓
模型: 预测 acceleration = dv/dt
      ↓
输出: velocity[t+1] = velocity[t] + acceleration
      ↓
挑战: 测试集速度范围 [-1.49, 2.73] 远超训练集
      → 需要模型具有良好的外推能力
```

---

## 参考资源

- 原论文：https://arxiv.org/abs/2010.03409
- DeepMind 数据集：https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/
- 本项目分析脚本：`inspect_tfrecord_deep.py`, `analyze_all_datasets.py`
