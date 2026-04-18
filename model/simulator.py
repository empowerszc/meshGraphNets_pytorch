"""
Simulator 模块 - MeshGraphNet 物理模拟器

这是整个模型的最高层封装，负责:
1. 管理三个归一化器 (节点、边、输出)
2. 构造节点特征 (速度 + 节点类型 one-hot)
3. 训练/推理模式切换
4. 噪声注入 (训练技巧)
5. 加速度 ↔ 速度转换

与论文的关系:
- 对应 "Learning Mesh-Based Simulation with Graph Networks" 的完整训练/推理流程
"""

import torch.nn.init as init
import torch.nn as nn
import torch
from torch_geometric.data import Data

from .model import EncoderProcesserDecoder
from utils import normalization


def init_weights(m):
    """
    权重初始化：Xavier 均匀初始化

    作用：
    - 使初始权重服从合理的分布
    - 避免梯度消失/爆炸
    - 加速模型收敛
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class Simulator(nn.Module):
    """
    物理模拟器主类

    封装了 Encoder-Processor-Decoder 模型，并添加了:
    - 特征归一化 (在线统计累积)
    - 节点特征构造 (速度 + 节点类型)
    - 训练时噪声注入
    - 推理时速度积分

    参数:
        message_passing_num: 消息传递层数 (默认 15)
        node_input_size: 节点输入维度 (11 = 2 速度 + 9 节点类型)
        edge_input_size: 边输入维度 (3 = 2 相对坐标 + 1 距离)
        device: 运行设备 ('cuda:0' 或 'cpu')

    输入输出关系:
        训练：(图，噪声) → (预测加速度，目标加速度) → MSE Loss
        推理：(图，None) → 下一时刻速度预测
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        device: str,
    ) -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

        # 核心 GNN 模型
        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            edge_input_size=edge_input_size
        ).to(device)

        # 三个归一化器：分别处理输出、节点特征、边特征
        # 使用在线统计累积，无需预先扫描数据集
        self._output_normalizer = normalization.Normalizer(
            size=2, name='output_normalizer', device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=node_input_size, name='node_normalizer', device=device
        )
        self.edge_normalizer = normalization.Normalizer(
            size=edge_input_size, name='edge_normalizer', device=device
        )

        # Xavier 初始化所有权重
        self.model.apply(init_weights)
        print('Simulator model initialized')

    def update_node_attr(self, frames: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """
        构造并归一化节点特征

        Args:
            frames: [N, 2] — 当前速度 (或带噪声的速度，训练时)
            types: [N, 1] — 节点类型索引

        Returns:
            归一化后的节点特征 [N, 11]

        特征构造过程:
            1. 节点类型 → one-hot 编码 [N, 9]
            2. 拼接：velocity[N, 2] + one_hot[N, 9] = [N, 11]
            3. 归一化：使用在线累积的均值和标准差

        节点类型 (9 种):
            0-NORMAL, 1-OBSTACLE, 2-AIRFOIL, 3-HANDLE,
            4-INFLOW, 5-OUTFLOW, 6-WALL_BOUNDARY
        """
        node_type = types.squeeze(-1).long()  # [N]
        one_hot = torch.nn.functional.one_hot(node_type, num_classes=9)  # [N, 9]
        node_feats = torch.cat([frames, one_hot], dim=-1)  # [N, 2 + 9 = 11]
        normalized_feats = self._node_normalizer(node_feats, self.training)
        return normalized_feats

    @staticmethod
    def velocity_to_acceleration(noised_frames: torch.Tensor, next_velocity: torch.Tensor) -> torch.Tensor:
        """
        速度 → 加速度转换

        物理意义：加速度 = 速度变化率
        a(t) = v(t+1) - v(t)

        Args:
            noised_frames: [N, 2] — 当前时刻速度 (可能带噪声)
            next_velocity: [N, 2] — 下一时刻目标速度

        Returns:
            acceleration: [N, 2] — 目标加速度

        为什么预测加速度而不是速度？
        - 加速度通常比速度更小、更平滑
        - 便于归一化处理
        - 符合物理方程的形式 (F=ma)
        """
        return next_velocity - noised_frames

    def forward(self, graph: Data, velocity_sequence_noise: torch.Tensor):
        """
        模拟器前向传播

        ┌─────────────────────────────────────────────────────────────┐
        │                    Training Mode                            │
        ├─────────────────────────────────────────────────────────────┤
        │ 1. 读取速度 frames 和节点类型                                │
        │ 2. 注入噪声：noised_frames = frames + noise                 │
        │ 3. 构造节点特征：[noised_frames, one_hot_type]              │
        │ 4. GNN 前向传播 → predicted_acc_norm                        │
        │ 5. 计算目标加速度：target_acc = next_v - noised_frames      │
        │ 6. 归一化目标加速度 → target_acc_norm                       │
        │ 7. 返回：(predicted_acc_norm, target_acc_norm)              │
        └─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                   Inference Mode                            │
        ├─────────────────────────────────────────────────────────────┤
        │ 1. 读取速度 frames 和节点类型                                │
        │ 2. 构造节点特征：[clean_frames, one_hot_type]               │
        │ 3. GNN 前向传播 → predicted_acc_norm                        │
        │ 4. 反归一化 → acc_update                                    │
        │ 5. 速度积分：predicted_v = frames + acc_update              │
        │ 6. 返回：predicted_velocity                                 │
        └─────────────────────────────────────────────────────────────┘

        Args:
            graph: PyG Data 对象
                - graph.x[:, 0:1]: 节点类型 [N, 1]
                - graph.x[:, 1:3]: 当前速度 [N, 2]
                - graph.edge_attr: 边特征 [E, 3]
                - graph.y: 下一时刻目标速度 [N, 2] (仅训练时需要)
            velocity_sequence_noise: 训练时注入的速度噪声 [N, 2]
                                   推理时为 None

        Returns:
            训练模式：(predicted_acc_norm, target_acc_norm)
            推理模式：predicted_velocity [N, 2]
        """
        node_type = graph.x[:, 0:1]      # [N, 1]
        frames = graph.x[:, 1:3]         # [N, 2] — current velocity

        if self.training:
            # ========== 训练模式 ==========
            assert velocity_sequence_noise is not None, "Noise must be provided during training"

            # 注入噪声 (提高鲁棒性)
            noised_frames = frames + velocity_sequence_noise  # [N, 2]

            # 构造归一化节点特征
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr

            # 归一化边特征
            edge_attr = graph.edge_attr  # [E, 3]
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr

            # GNN 前向传播 → 预测归一化加速度
            predicted_acc_norm = self.model(graph)  # [N, 2]

            # 计算目标加速度并归一化
            target_vel = graph.y  # [N, 2]
            target_acc = self.velocity_to_acceleration(noised_frames, target_vel)
            target_acc_norm = self._output_normalizer(target_acc, self.training)

            # 返回预测和目标 (用于计算 MSE Loss)
            return predicted_acc_norm, target_acc_norm

        else:
            # ========== 推理模式 ==========
            # 保存原始 graph.x 用于恢复
            original_x = graph.x.clone()

            # 使用干净的速度 (无噪声)
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr

            # 归一化边特征
            edge_attr = graph.edge_attr  # [E, 3]
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr

            # GNN 前向传播 → 预测归一化加速度
            predicted_acc_norm = self.model(graph)  # [N, 2]

            # 反归一化得到实际加速度
            acc_update = self._output_normalizer.inverse(predicted_acc_norm)  # [N, 2]

            # 速度积分：v(t+1) = v(t) + a(t)
            predicted_velocity = frames + acc_update

            # 恢复原始 graph.x (保留 node_type 用于下一次 rollout)
            graph.x = original_x

            return predicted_velocity