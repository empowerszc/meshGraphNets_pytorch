"""
可视化理解 MeshGraphNet 的数据结构和模型消费过程

这个问题非常核心：模型每一时刻预测的是什么？节点、网点、三角网格是如何被消费的？
"""

import tensorflow as tf
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 启用 eager execution
tf.compat.v1.enable_eager_execution()

DATA_DIR = "/Volumes/seagate/300Learn/310CodePractice/bytime/0417_meshgraphnet_understand/meshGraphNets_pytorch/data"

# ==================== 第一部分：读取一条数据 ====================

def load_trajectory(file_path, index=0):
    """加载一条轨迹数据"""
    dataset = tf.data.TFRecordDataset(file_path)
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

            pressure = np.frombuffer(
                features['pressure'].bytes_list.value[0],
                dtype=np.float32
            ).reshape(-1, mesh_pos.shape[0])

            return {
                'mesh_pos': mesh_pos,
                'node_type': node_type,
                'cells': cells,
                'velocity': velocity,  # [T, N, 2]
                'pressure': pressure   # [T, N]
            }
    return None


# ==================== 第二部分：详细解释每种数据 ====================

def explain_data_types(data):
    """详细解释每种数据的含义"""

    print("\n" + "="*80)
    print(" " * 25 + "MeshGraphNet 数据结构详解")
    print("="*80)

    # 1. 网格节点 (mesh_pos)
    print("\n【1】网格节点位置 (mesh_pos)")
    print("-" * 60)
    print(f"形状：{data['mesh_pos'].shape}")
    print(f"含义：每个节点在 2D 空间中的 (x, y) 坐标")
    print(f"节点总数：{data['mesh_pos'].shape[0]}")
    print(f"\n前 5 个节点位置:")
    for i in range(5):
        print(f"  节点{i}: ({data['mesh_pos'][i, 0]:.4f}, {data['mesh_pos'][i, 1]:.4f})")

    print("\n物理意义:")
    print("  这些点构成了流体计算域的离散化网格。每个点是空间中的一个固定位置，")
    print("  模型在这些点上预测流速。网格是不均匀的——关键区域（如圆柱表面）更密集。")

    # 2. 节点类型 (node_type)
    print("\n【2】节点类型 (node_type)")
    print("-" * 60)
    print(f"形状：{data['node_type'].shape}")
    print(f"含义：每个节点的物理属性分类")
    print(f"唯一值：{np.unique(data['node_type'])}")

    type_names = {0: 'NORMAL', 4: 'INFLOW', 5: 'OUTFLOW', 6: 'WALL'}
    print(f"\n节点类型分布:")
    for t in np.unique(data['node_type']):
        count = np.sum(data['node_type'] == t)
        pct = count / len(data['node_type']) * 100
        print(f"  {type_names.get(t, f'UNKNOWN_{t}')}: {count} ({pct:.1f}%)")

    print("\n物理意义:")
    print("  - NORMAL (0): 普通流体区域，模型需要预测这些点的速度变化")
    print("  - INFLOW (4): 入口边界，速度由外部条件给定（通常固定）")
    print("  - OUTFLOW (5): 出口边界，模型预测（允许流体自由流出）")
    print("  - WALL (6): 固壁边界，速度恒为零（无滑移条件）")

    # 3. 三角网格 (cells)
    print("\n【3】三角网格单元 (cells)")
    print("-" * 60)
    print(f"形状：{data['cells'].shape}")
    print(f"含义：三角单元连接关系，每个单元由 3 个节点索引组成")
    print(f"三角单元总数：{data['cells'].shape[0]}")
    print(f"\n前 5 个三角单元:")
    for i in range(5):
        print(f"  单元{i}: 节点[{data['cells'][i, 0]}, {data['cells'][i, 1]}, {data['cells'][i, 2]}]")

    print("\n物理意义:")
    print("  三角网格定义了节点之间的连接关系（拓扑结构）。在图神经网络中，")
    print("  这被转换为边 (edge)——如果两个节点在同一个三角形中，它们之间就有边。")
    print("  消息传递沿这些边进行。")

    # 4. 速度场 (velocity)
    print("\n【4】速度场 (velocity)")
    print("-" * 60)
    print(f"形状：{data['velocity'].shape}")
    print(f"含义：每个节点在每个时间步的 2D 速度向量 (vx, vy)")
    print(f"时间步数：{data['velocity'].shape[0]}")
    print(f"速度范围：[{data['velocity'].min():.4f}, {data['velocity'].max():.4f}]")
    print(f"\nt=0 时刻的速度示例 (前 5 个节点):")
    for i in range(5):
        vx, vy = data['velocity'][0, i]
        print(f"  节点{i}: vx={vx:.4f}, vy={vy:.4f}, 速度大小={np.sqrt(vx**2+vy**2):.4f}")

    print("\n物理意义:")
    print("  速度场是模型要预测的核心物理量。在每个时间步 t，模型观察当前速度 v(t)，")
    print("  预测下一时刻的速度 v(t+1)。这是欧拉描述——在固定空间点上观察流体运动。")

    # 5. 压力场 (pressure)
    print("\n【5】压力场 (pressure) [辅助物理量]")
    print("-" * 60)
    print(f"形状：{data['pressure'].shape}")
    print(f"含义：每个节点在每个时间步的压力（标量场）")
    print(f"压力范围：[{data['pressure'].min():.4f}, {data['pressure'].max():.4f}]")
    print("\n注意：本项目中模型只预测速度，不预测压力。")

    print("\n" + "="*80)


# ==================== 第三部分：模型如何消费这些数据 ====================

def explain_model_consumption(data):
    """解释模型如何消费这些数据"""

    print("\n" + "="*80)
    print(" " * 20 + "模型消费数据的过程详解")
    print("="*80)

    print("""
【核心问题】模型每一时刻预测的是什么？

答案：模型预测的是 加速度 (acceleration)，即速度的变化率 dv/dt。

为什么预测加速度而不是直接预测速度？
  1. 加速度通常比速度更小、更平滑，便于神经网络学习
  2. 符合物理方程的形式（F = ma，力产生加速度）
  3. 通过积分 v(t+1) = v(t) + a(t) 得到速度，数值更稳定
  4. 便于使用归一化器处理

训练时:
  输入：当前速度 v(t) + 噪声
  预测：归一化的加速度 a_norm
  目标：归一化的 (v(t+1) - v(t))
  Loss: MSE(a_norm, target_norm)

推理时:
  输入：当前速度 v(t)
  预测：a_norm → 反归一化 → a(t)
  输出：v(t+1) = v(t) + a(t)
""")

    print("\n【详细消费流程】")
    print("-" * 60)

    print("""
Step 1: 构造输入特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每个节点的输入特征 = [node_type_onehot (9 维), velocity (2 维)]
                   = 11 维向量

例如节点 i 的特征：
  node_type = 0 (NORMAL)
  velocity  = [0.5, 0.1]

  x[i] = [1,0,0,0,0,0,0,0,0,  0.5,0.1]
         └──── node_type ────┘ └─ vel ─┘
              (9 维)              (2 维)

为什么 node_type 要 one-hot？
  → 让模型能够区分不同物理意义的节点
""")

    print("""
Step 2: 构建图结构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：cells [num_cells, 3] (三角网格)
      ↓ FaceToEdge 变换
输出：edge_index [2, num_edges]

边的生成规则:
  每个三角形 (i, j, k) 生成 3 条边：
    - i → j
    - j → k
    - k → i
  (也可能生成双向边，取决于实现)

边特征 (edge_attr):
  edge_attr = [dx, dy, distance]
            = [x_j-x_i, y_j-y_i, sqrt(dx²+dy²)]
  维度：3 维

物理意义：
  - dx, dy: 相对位置编码（平移不变性）
  - distance: 节点间距离（影响消息传递权重）
""")

    print("""
Step 3: Encoder 编码
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
节点特征：x [N, 11] → MLP → h [N, 128]
边特征：  e [E, 3]  → MLP → e' [E, 128]

目的：将原始特征映射到统一的隐空间 (128 维)
""")

    print("""
Step 4: Processor 消息传递 (×15 层 GnBlock)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每一层 GnBlock 执行:

┌─────────────────────────────────────────────────┐
│ EdgeBlock (边更新):                              │
│ ───────────────────                              │
│ 对于每条边 i→j:                                  │
│ e'ᵢⱼ = MLPₑ([hᵢ, hⱼ, eᵢⱼ])                     │
│                                                │
│ 其中：hᵢ = sender 节点特征                       │
│        hⱼ = receiver 节点特征                   │
│        eᵢⱼ = 当前边特征                          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ NodeBlock (节点更新):                            │
│ ────────────────────                             │
│ 对于每个节点 j:                                  │
│ mⱼ = Σᵢ∈N(j) e'ᵢⱼ   (聚合所有入边)            │
│ h'ⱼ = MLPₙ([hⱼ, mⱼ])                           │
│                                                │
│ 其中：N(j) = 所有以 j 为目标的 sender 节点集合    │
│        mⱼ = 聚合的消息                          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 残差连接:                                        │
│ ─────────                                        │
│ hⱼ ← hⱼ + h'ⱼ                                  │
│ eᵢⱼ ← eᵢⱼ + e'ᵢⱼ                              │
└─────────────────────────────────────────────────┘

15 层 GnBlock 的意义:
  - 每层聚合 1 跳邻域信息
  - 15 层后，每个节点的感受野 ≈ 15 跳
  - 对于典型网格，这足以覆盖整个计算域
""")

    print("""
Step 5: Decoder 输出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h [N, 128] → MLP → a [N, 2]

输出：a = [ax, ay] = 加速度向量
      即每个节点的速度变化率预测

为什么是 2 维？
  → 2D 流场，每个节点有 x 和 y 方向的加速度
""")

    print("""
Step 6: 速度积分
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v(t+1) = v(t) + a(t)

这就是物理学中的欧拉积分：
  新速度 = 旧速度 + 加速度 × 时间步长

注意：
  - 这里假设时间步长为 1（归一化单位）
  - 实际物理时间需要根据具体场景缩放
""")

    print("\n" + "="*80)


# ==================== 第四部分：可视化 ====================

def visualize_mesh_and_types(data):
    """可视化网格和节点类型"""

    if not HAS_MATPLOTLIB:
        print("\n⚠️  未安装 matplotlib，跳过可视化")
        print("   安装：pip install matplotlib")
        return

    import matplotlib.tri as tri

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 网格可视化
    ax = axes[0, 0]
    triang = tri.Triangulation(data['mesh_pos'][:, 0], data['mesh_pos'][:, 1], triangles=data['cells'])
    ax.triplot(triang, 'b-', alpha=0.3, lw=0.5)
    ax.set_title('三角网格结构', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # 2. 节点类型可视化
    ax = axes[0, 1]
    type_names = {0: 'NORMAL', 4: 'INFLOW', 5: 'OUTFLOW', 6: 'WALL'}
    colors = {0: '#1f77b4', 4: '#2ca02c', 5: '#d62728', 6: '#7f7f7f'}

    for t, name in type_names.items():
        mask = (data['node_type'] == t)
        if np.any(mask):
            ax.scatter(data['mesh_pos'][mask, 0], data['mesh_pos'][mask, 1],
                      c=[colors[t]], label=name, s=30, edgecolors='black', linewidth=0.5)

    ax.triplot(triang, 'k-', alpha=0.1, lw=0.3)
    ax.legend()
    ax.set_title('节点类型分布', fontsize=12)
    ax.set_aspect('equal')

    # 3. 速度场可视化 (t=0)
    ax = axes[1, 0]
    speed = np.linalg.norm(data['velocity'][0], axis=-1)
    cf = ax.tripcolor(triang, speed, shading='gouraud', cmap='viridis')
    ax.triplot(triang, 'k-', alpha=0.2, lw=0.3)
    plt.colorbar(cf, ax=ax, label='速度大小')
    ax.set_title('t=0 时刻速度场', fontsize=12)
    ax.set_aspect('equal')

    # 4. 速度场可视化 (t=300)
    ax = axes[1, 1]
    speed = np.linalg.norm(data['velocity'][300], axis=-1)
    cf = ax.tripcolor(triang, speed, shading='gouraud', cmap='viridis')
    ax.triplot(triang, 'k-', alpha=0.2, lw=0.3)
    plt.colorbar(cf, ax=ax, label='速度大小')
    ax.set_title('t=300 时刻速度场', fontsize=12)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(DATA_DIR), 'data_types_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到：{save_path}")
    plt.show()


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("加载训练集第一条轨迹...")
    data = load_trajectory(os.path.join(DATA_DIR, "train.tfrecord"), index=0)

    if data:
        print(f"✓ 成功加载数据")
        print(f"  mesh_pos: {data['mesh_pos'].shape}")
        print(f"  node_type: {data['node_type'].shape}")
        print(f"  cells: {data['cells'].shape}")
        print(f"  velocity: {data['velocity'].shape}")
        print(f"  pressure: {data['pressure'].shape}")

        # 解释数据类型
        explain_data_types(data)

        # 解释模型消费过程
        explain_model_consumption(data)

        # 可视化
        print("\n生成可视化图表...")
        visualize_mesh_and_types(data)
    else:
        print("加载失败!")
