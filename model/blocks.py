import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import Data


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
       
            
            
        