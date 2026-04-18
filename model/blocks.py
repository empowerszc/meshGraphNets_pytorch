import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import Data


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    """
    构建多层感知机 (MLP)

    结构：Linear → ReLU → Linear → ReLU → Linear → ReLU → Linear → [LayerNorm]
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


class EdgeBlock(nn.Module):
    """
    边更新模块 (Edge Update Block)

    功能：根据边两端的节点特征和边自身的特征，更新边特征。
    这是 Graph Network 框架中的 EdgeBlock，对应论文中的 Edge MLP。

    输入图结构:
        - node_attr: [N, node_dim] 节点特征
        - edge_attr: [E, edge_dim] 边特征
        - edge_index: [2, E] 边索引 (senders, receivers)

    输出:
        - 更新后的边特征 [E, hidden_dim]
    """

    def __init__(self, custom_func: nn.Module):
        """
        Args:
            custom_func: 自定义的 MLP 函数，用于处理拼接后的边特征
                        输入维度 = 2 * node_dim + edge_dim
                        输出维度 = hidden_dim
        """
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        """
        前向传播：执行边特征更新

        步骤:
        1. 获取发送者和接收者节点的索引
        2. 收集发送者和接收者的节点特征
        3. 拼接 [sender_feat, receiver_feat, edge_feat]
        4. 通过 MLP 得到新的边特征
        """
        node_attr = graph.x  # [N, node_dim]
        senders_idx, receivers_idx = graph.edge_index  # [E], [E]
        edge_attr = graph.edge_attr  # [E, edge_dim]

        edges_to_collect = []

        # 获取发送者节点特征 [E, node_dim]
        senders_attr = node_attr[senders_idx]
        # 获取接收者节点特征 [E, node_dim]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        # 拼接后的边特征 [E, 2*node_dim + edge_dim]
        collected_edges = torch.cat(edges_to_collect, dim=1)

        # 通过 MLP 更新边特征 [E, hidden_dim]
        edge_attr = self.net(collected_edges)

        return Data(x=node_attr, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):
    """
    节点更新模块 (Node Update Block)

    功能：聚合与节点相连的边信息，并更新节点特征。
    这是 Graph Network 框架中的 NodeBlock，对应论文中的 Node MLP。

    核心操作:
        1. Scatter Add: 将所有入边的特征聚合到接收节点
        2. 拼接：原节点特征 + 聚合的边信息
        3. 通过 MLP 得到新的节点特征

    输入图结构:
        - node_attr: [N, node_dim] 节点特征
        - edge_attr: [E, edge_dim] 边特征 (已由 EdgeBlock 更新)
        - edge_index: [2, E] 边索引

    输出:
        - 更新后的节点特征 [N, hidden_dim]
    """

    def __init__(self, custom_func: nn.Module):
        """
        Args:
            custom_func: 自定义的 MLP 函数，用于处理拼接后的节点特征
                        输入维度 = node_dim + edge_dim
                        输出维度 = hidden_dim
        """
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        """
        前向传播：执行节点特征更新

        步骤:
        1. 获取接收者节点索引 (每条边的目标节点)
        2. 使用 scatter_add 将边特征聚合到节点
           - 例如：节点 i 的聚合特征 = sum(所有以 i 为接收者的边特征)
        3. 拼接 [原节点特征，聚合的边特征]
        4. 通过 MLP 得到新的节点特征
        """
        # Decompose graph
        edge_attr = graph.edge_attr  # [E, edge_dim]
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index  # [E] 接收者节点索引
        num_nodes = graph.num_nodes

        # 关键操作：将边特征按接收者节点索引进行累加聚合
        # scatter_add 的作用：对于每个节点，累加所有以它为接收者的边特征
        # 输出形状：[N, edge_dim]
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)  # 原始节点特征 [N, node_dim]
        nodes_to_collect.append(agg_received_edges)  # 聚合的边特征 [N, edge_dim]

        # 拼接后的节点特征 [N, node_dim + edge_dim]
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        # 通过 MLP 更新节点特征 [N, hidden_dim]
        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


# ============================================================================
# ONNX 导出兼容版本
# ============================================================================

def scatter_add_onnx(edge_attr: torch.Tensor, receivers_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    ONNX 兼容的 scatter_add 实现

    由于 torch_scatter.scatter_add 在 ONNX 导出时有限制，
    我们使用纯 PyTorch 操作实现相同的功能。

    Args:
        edge_attr: [E, edge_dim] 边特征
        receivers_idx: [E] 接收者节点索引
        num_nodes: 节点总数

    Returns:
        [N, edge_dim] 聚合后的节点特征
    """
    # 创建零张量用于累加 [N, edge_dim]
    result = torch.zeros(num_nodes, edge_attr.shape[1], device=edge_attr.device, dtype=edge_attr.dtype)

    # 使用 scatter_add_ 进行原地累加
    # 这比 for 循环高效得多，且可以导出为 ONNX
    result.scatter_add_(0, receivers_idx.unsqueeze(1).expand_as(edge_attr), edge_attr)

    return result


class EdgeBlockONNX(nn.Module):
    """
    ONNX 兼容的 EdgeBlock

    功能与原 EdgeBlock 相同，但避免了 PyG 特定的操作，
    直接使用张量而不是 Data 对象。
    """

    def __init__(self, custom_func: nn.Module):
        super(EdgeBlockONNX, self).__init__()
        self.net = custom_func

    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        """
        ONNX 兼容的前向传播

        Args:
            node_attr: [N, node_dim] 节点特征
            edge_attr: [E, edge_dim] 边特征
            edge_index: [2, E] 边索引

        Returns:
            更新后的边特征 [E, hidden_dim]
        """
        senders_idx, receivers_idx = edge_index  # [E], [E]

        # 获取发送者和接收者节点特征
        senders_attr = node_attr[senders_idx]  # [E, node_dim]
        receivers_attr = node_attr[receivers_idx]  # [E, node_dim]

        # 拼接 [sender, receiver, edge] -> [E, 2*node_dim + edge_dim]
        collected_edges = torch.cat([senders_attr, receivers_attr, edge_attr], dim=1)

        # 通过 MLP 更新边特征
        updated_edge_attr = self.net(collected_edges)

        return updated_edge_attr


class NodeBlockONNX(nn.Module):
    """
    ONNX 兼容的 NodeBlock

    功能与原 NodeBlock 相同，但使用纯 PyTorch 操作。
    """

    def __init__(self, custom_func: nn.Module):
        super(NodeBlockONNX, self).__init__()
        self.net = custom_func

    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        """
        ONNX 兼容的前向传播

        Args:
            node_attr: [N, node_dim] 节点特征
            edge_attr: [E, edge_dim] 边特征
            edge_index: [2, E] 边索引
            num_nodes: 节点总数 N

        Returns:
            更新后的节点特征 [N, hidden_dim]
        """
        _, receivers_idx = edge_index  # [E]

        # 聚合入边特征到节点
        agg_edges = scatter_add_onnx(edge_attr, receivers_idx, num_nodes)  # [N, edge_dim]

        # 拼接 [原节点特征，聚合边特征] -> [N, node_dim + edge_dim]
        collected_nodes = torch.cat([node_attr, agg_edges], dim=-1)

        # 通过 MLP 更新节点特征
        updated_node_attr = self.net(collected_nodes)

        return updated_node_attr


class GnBlockONNX(nn.Module):
    """
    ONNX 兼容的 GnBlock (Graph Network Block)

    包含 EdgeBlock 和 NodeBlock，以及残差连接。
    这是 ONNX 导出的基本构建块。
    """

    def __init__(self, hidden_size: int = 128):
        super(GnBlockONNX, self).__init__()

        # 构建 EdgeBlock 和 NodeBlock 的 MLP
        eb_input_dim = 3 * hidden_size  # sender + receiver + edge
        nb_input_dim = 2 * hidden_size  # node + aggregated_edge

        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlockONNX(custom_func=eb_custom_func)
        self.nb_module = NodeBlockONNX(custom_func=nb_custom_func)

    def forward(self, node_attr: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        """
        ONNX 兼容的前向传播

        Args:
            node_attr: [N, node_dim] 节点特征
            edge_attr: [E, edge_dim] 边特征
            edge_index: [2, E] 边索引
            num_nodes: 节点总数 N

        Returns:
            (updated_node_attr, updated_edge_attr) 元组
        """
        # 保存残差连接
        residual_node = node_attr
        residual_edge = edge_attr

        # EdgeBlock: 更新边特征
        updated_edge = self.eb_module(node_attr, edge_attr, edge_index)

        # NodeBlock: 更新节点特征
        updated_node = self.nb_module(node_attr, updated_edge, edge_index, num_nodes)

        # 残差连接
        updated_node = updated_node + residual_node
        updated_edge = updated_edge + residual_edge

        return updated_node, updated_edge
