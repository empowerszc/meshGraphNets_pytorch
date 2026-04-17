"""
Noise - 噪声注入工具

本模块实现了训练时的速度噪声注入功能，这是一种正则化技术，
用于提高模型的鲁棒性和长期稳定性。

为什么需要噪声注入？
──────────────────────────────────────────────────────────────
1. 提高鲁棒性：使模型对输入扰动不敏感
2. 模拟数值误差：传统 CFD 求解器存在截断误差和舍入误差
3. 防止过拟合：作为一种数据增强手段
4. 改善长期稳定性：自回归推理时误差累积更慢

为什么只在 NORMAL 节点加噪？
──────────────────────────────────────────────────────────────
- 边界节点 (INFLOW, WALL 等) 的速度由边界条件固定
- 这些节点的速度在 rollout 时会被强制重置
- 给它们加噪没有意义，反而可能干扰模型学习边界条件
"""

import torch
from utils.utils import NodeType


def get_velocity_noise(graph, noise_std, device):
    """
    生成速度序列噪声

    功能：
    1. 生成高斯噪声
    2. 仅在 NORMAL 节点上应用噪声
    3. 边界节点噪声置零

    参数:
        graph: PyG Data 对象
            - graph.x[:, 0]: 节点类型 [N]
            - graph.x[:, 1:3]: 速度 [N, 2]
        noise_std: 噪声标准差 (默认 2e-2)
        device: 运行设备

    Returns:
        noise: 速度噪声 [N, 2]
            - NORMAL 节点：高斯噪声 ~ N(0, noise_std²)
            - 其他节点：0 (无噪声)

    噪声生成过程:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 生成独立同分布的高斯噪声 [N, 2]                       │
    │    noise ~ N(0, noise_std²)                             │
    │                                                         │
    │ 2. 创建掩码：mask = (node_type != NORMAL)               │
    │                                                         │
    │ 3. 掩码置零：noise[mask] = 0                            │
    │                                                         │
    │ 结果：仅流体区域的节点有噪声，边界条件保持干净           │
    └─────────────────────────────────────────────────────────┘

    示例:
        >>> noise = get_velocity_noise(graph, noise_std=0.02, device='cuda')
        >>> noised_velocity = graph.x[:, 1:3] + noise
    """
    # 提取当前速度 [N, 2]
    velocity_sequence = graph.x[:, 1:3]
    # 提取节点类型 [N]
    node_type = graph.x[:, 0]

    # 生成高斯噪声：每个元素独立采样自 N(0, noise_std²)
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)

    # 创建掩码：非 NORMAL 节点 (边界、障碍物等)
    mask = (node_type != NodeType.NORMAL)

    # 边界节点的噪声置零
    noise[mask] = 0

    return noise.to(device)
