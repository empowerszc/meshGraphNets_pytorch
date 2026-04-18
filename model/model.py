"""
MeshGraphNet 核心模型架构

实现 Encoder-Processor-Decoder 三段式架构:
┌─────────────────────────────────────────────────────────────┐
│                      MeshGraphNet                           │
├─────────────────────────────────────────────────────────────┤
│  Encoder    → 特征映射到隐空间 (128 维)                      │
│      ↓                                                      │
│  GnBlock ×15 → 消息传递 (边更新 + 节点更新 + 残差)            │
│      ↓                                                      │
│  Decoder    → 输出预测 (2D 加速度)                            │
└─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock, EdgeBlockONNX, NodeBlockONNX, GnBlockONNX
from torch_geometric.data import Data


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    """
    构建多层感知机 (MLP)

    结构：Linear → ReLU → Linear → ReLU → Linear → ReLU → Linear → [LayerNorm]

    Args:
        in_size: 输入维度
        hidden_size: 隐藏层维度
        out_size: 输出维度
        lay_norm: 是否在输出层添加 LayerNorm

    Returns:
        nn.Sequential: 构建好的 MLP 模块

    示例:
        build_mlp(128, 128, 2) → 输入 128 → 输出 2 的 MLP
    """
    module = nn.Sequential(
        nn.Linear(in_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):
    """
    编码器 (Encoder)

    功能：将原始的节点特征和边特征映射到统一的隐空间表示。

    结构:
        - 节点编码器：MLP(node_input_size → hidden_size)
        - 边编码器：MLP(edge_input_size → hidden_size)

    输入:
        - graph.x: [N, node_input_size] 原始节点特征 (速度 + 节点类型 one-hot)
        - graph.edge_attr: [E, edge_input_size] 原始边特征 (相对坐标 + 距离)

    输出:
        - graph.x: [N, hidden_size] 编码后的节点特征
        - graph.edge_attr: [E, hidden_size] 编码后的边特征
    """

    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super(Encoder, self).__init__()

        # 边特征编码器 MLP
        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        # 节点特征编码器 MLP
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):
        """
        前向传播：分别对节点和边特征进行编码

        注意：这里只是简单的线性映射，不涉及图结构的信息传递
        """
        node_attr, edge_attr = graph.x, graph.edge_attr

        # 节点特征编码 [N, hidden_size]
        node_ = self.nb_encoder(node_attr)
        # 边特征编码 [E, hidden_size]
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class EncoderONNX(nn.Module):
    """
    ONNX 兼容的编码器

    功能与原 Encoder 相同，但直接处理张量而非 Data 对象。
    """

    def __init__(self, edge_input_size=3, node_input_size=11, hidden_size=128):
        super(EncoderONNX, self).__init__()

        # 边特征编码器 MLP
        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        # 节点特征编码器 MLP
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor):
        """
        ONNX 兼容的前向传播

        Args:
            node_attr: [N, node_input_size] 节点特征
            edge_attr: [E, edge_input_size] 边特征

        Returns:
            (encoded_node_attr, encoded_edge_attr) 元组
        """
        encoded_node = self.nb_encoder(node_attr)
        encoded_edge = self.eb_encoder(edge_attr)
        return encoded_node, encoded_edge


class GnBlock(nn.Module):
    """
    图网络基本块 (Graph Network Block)

    这是 MeshGraphNet 的核心处理单元，包含：
    1. EdgeBlock: 更新边特征
    2. NodeBlock: 更新节点特征
    3. 残差连接：将输入加到输出上，促进深层网络训练

    对应论文中的 "Multi-step Graph Neural Network" 部分。

    结构:
        输入 → EdgeBlock → NodeBlock → 残差连接 → 输出

    维度变化 (假设 hidden_size=128):
        - 输入节点：[N, 128]
        - 输入边：[E, 128]
        - EdgeBlock 输入：[E, 384] (128*2 + 128，拼接 sender/receiver/edge)
        - NodeBlock 输入：[N, 256] (128 + 128，拼接 node/agg_edge)
    """

    def __init__(self, hidden_size=128):
        super(GnBlock, self).__init__()

        # EdgeBlock 输入维度 = 3 * hidden_size
        # 原因：sender_node_feat + receiver_node_feat + edge_feat
        eb_input_dim = 3 * hidden_size
        # NodeBlock 输入维度 = 2 * hidden_size
        # 原因：original_node_feat + aggregated_edge_feat
        nb_input_dim = 2 * hidden_size

        # 构建边更新和节点更新的 MLP
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        """
        前向传播：执行消息传递和残差连接

        步骤:
        1. 克隆输入 (保留残差连接的原始值)
        2. EdgeBlock 更新边特征
        3. NodeBlock 更新节点特征
        4. 残差连接：output = input + transformed_output

        残差连接的作用:
        - 缓解梯度消失，支持更深的网络 (15 层)
        - 学习残差函数 F(x) = H(x) - x，更容易优化
        """
        # 保存输入用于残差连接
        x = graph.x.clone()
        edge_attr = graph.edge_attr.clone()

        # 顺序处理：先边更新，后节点更新
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        # 残差连接
        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

    def forward_onnx(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        """
        ONNX 兼容的前向传播

        Args:
            node_attr: [N, hidden_size] 节点特征
            edge_attr: [E, hidden_size] 边特征
            edge_index: [2, E] 边索引
            num_nodes: 节点总数 N

        Returns:
            (updated_node_attr, updated_edge_attr) 元组
        """
        # 保存残差
        residual_node = node_attr
        residual_edge = edge_attr

        # 获取边索引
        senders_idx, receivers_idx = edge_index

        # === EdgeBlock ===
        # 获取发送者和接收者特征
        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]
        # 拼接并通过 MLP
        collected_edges = torch.cat([senders_attr, receivers_attr, edge_attr], dim=1)
        updated_edge = self.eb_module.net(collected_edges)

        # === NodeBlock ===
        # 聚合入边
        from .blocks import scatter_add_onnx
        agg_edges = scatter_add_onnx(updated_edge, receivers_idx, num_nodes)
        # 拼接并通过 MLP
        collected_nodes = torch.cat([node_attr, agg_edges], dim=-1)
        updated_node = self.nb_module.net(collected_nodes)

        # 残差连接
        updated_node = updated_node + residual_node
        updated_edge = updated_edge + residual_edge

        return updated_node, updated_edge


class Decoder(nn.Module):
    """
    解码器 (Decoder)

    功能：将处理后的节点特征映射到输出空间 (加速度预测)。

    结构：MLP(hidden_size → hidden_size → output_size)
    注意：解码器不使用 LayerNorm，因为输出不需要归一化

    输入:
        - graph.x: [N, hidden_size] 经过 15 层 GnBlock 处理后的节点特征

    输出:
        - [N, 2] 预测的加速度 (velocity_x, velocity_y)
    """

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        # lay_norm=False，输出层不需要归一化
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        """仅对节点特征进行解码，输出最终预测"""
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):
    """
    完整的 MeshGraphNet 模型

    架构：Encoder → [GnBlock × message_passing_num] → Decoder

    参数:
        message_passing_num: GnBlock 的重复次数，论文中使用 15 层
                           每一层执行一次消息传递 (边更新 + 节点更新)
        node_input_size: 节点输入特征维度 (11 = 2 速度 + 9 节点类型 one-hot)
        edge_input_size: 边输入特征维度 (3 = 2 相对坐标 + 1 距离)
        hidden_size: 隐空间维度，默认 128

    前向传播流程:
        1. Encoder: 将输入特征映射到 128 维隐空间
        2. GnBlock × 15: 执行 15 轮消息传递，聚合邻域信息
        3. Decoder: 输出 2D 加速度预测
    """

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):
        super(EncoderProcesserDecoder, self).__init__()

        # 编码器：特征映射
        self.encoder = Encoder(edge_input_size=edge_input_size,
                               node_input_size=node_input_size,
                               hidden_size=hidden_size)

        # 处理器：堆叠多个 GnBlock 进行消息传递
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        # 解码器：输出预测
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):
        """
        前向传播

        信息传播范围分析:
        - 1 层 GnBlock: 聚合 1 跳邻域信息
        - 15 层 GnBlock: 聚合 15 跳邻域信息
        对于典型网格，15 跳足以覆盖大部分局部相互作用
        """
        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded


class ONNXExportableMeshGraphNet(nn.Module):
    """
    可导出 ONNX 的 MeshGraphNet 版本

    由于 PyTorch Geometric 的某些操作 (如 scatter_add) 导出 ONNX 有限制，
    我们创建一个使用纯 PyTorch 操作的版本用于导出。

    主要特点:
    1. 直接使用张量而非 Data 对象
    2. 使用自定义的 scatter_add_onnx 替代 torch_scatter
    3. 所有维度显式传递，避免动态图结构推断

    输入:
        - node_attr: [N, 11] 节点特征 (速度 2 + 节点类型 one-hot 9)
        - edge_attr: [E, 3] 边特征 (dx, dy, distance)
        - edge_index: [2, E] 边索引

    输出:
        - [N, 2] 加速度预测
    """

    def __init__(self, message_passing_num=15, node_input_size=11, edge_input_size=3, hidden_size=128):
        super(ONNXExportableMeshGraphNet, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.message_passing_num = message_passing_num

        # 编码器
        self.node_encoder = build_mlp(node_input_size, hidden_size, hidden_size, lay_norm=True)
        self.edge_encoder = build_mlp(edge_input_size, hidden_size, hidden_size, lay_norm=True)

        # 处理器 - 使用 ONNX 兼容的 GnBlock
        self.gn_blocks = nn.ModuleList()
        for _ in range(message_passing_num):
            self.gn_blocks.append(GnBlockONNX(hidden_size=hidden_size))

        # 解码器
        self.decoder = build_mlp(hidden_size, hidden_size, 2, lay_norm=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        """
        前向传播

        Args:
            node_attr: [N, 11] 节点特征
            edge_attr: [E, 3] 边特征
            edge_index: [2, E] 边索引

        Returns:
            [N, 2] 加速度预测
        """
        N = node_attr.shape[0]

        # 编码
        node_feat = self.node_encoder(node_attr)
        edge_feat = self.edge_encoder(edge_attr)

        # 消息传递 (15 层)
        for block in self.gn_blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, edge_index, N)

        # 解码
        output = self.decoder(node_feat)

        return output
