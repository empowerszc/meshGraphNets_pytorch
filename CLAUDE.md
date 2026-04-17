# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目概述

MeshGraphNets 的 PyTorch + PyG 实现，用于学习基于网格的物理模拟。复现 DeepMind 论文 [Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)](https://arxiv.org/abs/2010.03409)，应用场景为圆柱绕流 (flow around a circular cylinder) 的流体动力学模拟。

## 快速开始命令

### 环境准备
```bash
pip install -r requirements.txt
```

### 数据准备
```bash
# 下载 DeepMind 数据集
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
aria2c -x 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data

# 转换数据格式 (需要 tensorflow<1.15)
python parse_tfrecord.py
```

### 训练
```bash
# 单 GPU 训练
python train.py

# 多 GPU 训练 (DDP)
export NGPUS=2
torchrun --nproc_per_node=$NGPUS train_ddp.py --dataset_dir data
```

### 推理与可视化
```bash
# 快速测试 (随机权重，无需训练)
python demo_inference.py --mode random --rollout_steps 10

# 加载模型推理
python demo_inference.py --mode checkpoint --checkpoint_path checkpoints/best_model.pth

# Rollout 评估
python rollout.py --rollout_num 5

# 渲染结果视频
python render_results.py
```

## 核心架构

### 数据流
```
TFRecord → parse_tfrecord.py → .npz + .dat
                              ↓
                    FpcDataset (memmap 加载)
                              ↓
                    PyG Data (x, pos, face, y)
                              ↓
          Transformer (FaceToEdge + Cartesian + Distance)
                              ↓
                    Simulator.forward()
                              ↓
                    Encoder → Processor(×15) → Decoder
```

### 模型组件

**Simulator** (`model/simulator.py`) - 最高层封装:
- 管理 3 个归一化器 (node/edge/output)
- 节点特征构造：`[velocity(2), node_type_onehot(9)] → [N, 11]`
- 训练模式：预测归一化加速度，返回 `(pred, target)` 用于 loss 计算
- 推理模式：预测下一时刻速度 `v(t+1) = v(t) + a(t)`

**EncoderProcesserDecoder** (`model/model.py`):
- **Encoder**: MLP 映射到隐空间 (node: 11→128, edge: 3→128)
- **GnBlock×15**: EdgeBlock → NodeBlock + 残差连接
- **Decoder**: MLP 输出加速度 (128→2)

**FpcDataset** (`dataset/fpc.py`):
- 使用 `np.memmap` 高效加载速度场数据
- 索引计算：`index // steps_per_trajectory` 定位轨迹

### 关键数据维度

| 张量 | 维度 | 说明 |
|------|------|------|
| `graph.x` | [N, 11] | 节点特征 (速度 2 + 节点类型 9) |
| `graph.pos` | [N, 2] | 节点位置 |
| `graph.edge_attr` | [E, 3] | 边特征 (dx, dy, distance) |
| `graph.face` | [3, F] | 三角面索引 |
| `graph.y` | [N, 2] | 目标速度 (下一时刻) |

### Loss 计算
仅在 `NORMAL` 和 `OUTFLOW` 节点上计算:
```python
mask = (node_type == NodeType.NORMAL) | (node_type == NodeType.OUTFLOW)
loss = MSE((pred - target)[mask])
```

## 节点类型

```python
class NodeType(enum.IntEnum):
    NORMAL = 0       # 普通流体区域 (预测)
    OBSTACLE = 1     # 障碍物 (固定)
    AIRFOIL = 2      # 翼型 (固定)
    HANDLE = 3       # 控制点 (固定)
    INFLOW = 4       # 入口 (固定)
    OUTFLOW = 5      # 出口 (预测)
    WALL_BOUNDARY = 6  # 壁面 (固定)
    SIZE = 9
```

## 训练技巧

1. **噪声注入**: 仅在 NORMAL 节点加噪 `noise_std=0.02`，提高鲁棒性
2. **在线归一化**: 累积统计量，无需预扫描数据集
3. **残差连接**: 支持 15 层深层网络训练
4. **边界约束**: rollout 时强制重置边界节点速度

## 文件结构

```
meshGraphNets_pytorch/
├── dataset/
│   └── fpc.py           # 数据集加载
├── model/
│   ├── blocks.py        # EdgeBlock, NodeBlock
│   ├── model.py         # Encoder-Processor-Decoder
│   └── simulator.py     # 主模型封装
├── utils/
│   ├── normalization.py # 在线归一化器
│   ├── noise.py         # 噪声注入
│   └── utils.py         # NodeType 枚举
├── train.py             # 单 GPU 训练
├── train_ddp.py         # 多 GPU DDP 训练
├── rollout.py           # 长时序推理
├── render_results.py    # 可视化渲染
├── parse_tfrecord.py    # TFRecord 解析
├── demo_inference.py    # 推理演示脚本
└── docs/
    └── understanding-notes.md  # 详细理解笔记
```

## 常见问题

**Q: Loss 为 NaN？**
- 检查学习率 (默认 1e-4)
- 检查归一化器状态：`model._output_normalizer._acc_count`

**Q: 如何调试消息传递？**
```python
def register_hooks(model):
    for i, block in enumerate(model.model.processer_list):
        block.nb_module.register_forward_hook(
            lambda m, inp, out: print(f"GnBlock{i} output: {out.mean().item():.4f}")
        )
```

## 相关文档
- `docs/understanding-notes.md` - 详细代码理解笔记，含数据流图、示例 Case、调试技巧
