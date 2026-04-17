"""
Utils - 工具类和常量定义

本模块定义了项目中使用的通用工具类和常量。
"""

import enum


class NodeType(enum.IntEnum):
    """
    节点类型枚举

    用于标识流场网格中不同位置的节点类型，
    每种类型对应不同的物理边界条件或区域属性。

    成员:
        NORMAL (0): 普通流体区域节点
                    - 流体自由流动的区域
                    - 模型需要预测其速度变化
                    - 训练和推理时的主要预测对象

        OBSTACLE (1): 障碍物节点
                    - 流场中的固定障碍物
                    - 速度恒为零 (无滑移边界条件)

        AIRFOIL (2): 翼型表面节点
                    - 类似圆柱表面的物体边界
                    - 速度恒为零 (无滑移边界条件)

        HANDLE (3): 控制点/手柄节点
                    - 用于几何变形的控制点
                    - 在某些变体中使用

        INFLOW (4): 入口边界节点
                    - 流体流入区域的边界
                    - 速度由入口条件给定 (通常固定)

        OUTFLOW (5): 出口边界节点
                    - 流体流出区域的边界
                    - 模型需要预测 (允许流体自由流出)
                    - 通常使用零梯度边界条件

        WALL_BOUNDARY (6): 壁面边界节点
                    - 管道或计算域的固壁边界
                    - 速度恒为零 (无滑移边界条件)

        SIZE (9): 节点类型总数
                  - 用于 one-hot 编码的类别数
                  - 注意：7 和 8  reserved 用于未来扩展

    节点类型与预测策略:
    ┌─────────────────────────────────────────────────────────┐
    │ 类型          │ 是否预测 │ 典型处理                       │
    ├─────────────────────────────────────────────────────────┤
    │ NORMAL       │ ✅ 是    │ 模型自由预测                    │
    │ OUTFLOW      │ ✅ 是    │ 模型自由预测                    │
    │ OBSTACLE     │ ❌ 否    │ 速度强制归零                    │
    │ AIRFOIL      │ ❌ 否    │ 速度强制归零                    │
    │ INFLOW       │ ❌ 否    │ 速度重置为入口条件              │
    │ WALL_BOUNDARY│ ❌ 否    │ 速度强制归零                    │
    └─────────────────────────────────────────────────────────┘

    在 Loss 计算中的应用:
    ```python
    mask = (node_type == NodeType.NORMAL) | (node_type == NodeType.OUTFLOW)
    loss = MSE(predicted[mask], target[mask])
    ```

    在噪声注入中的应用:
    ```python
    mask = (node_type != NodeType.NORMAL)
    noise[mask] = 0  # 仅 NORMAL 节点加噪
    ```

    示例:
        >>> NodeType.NORMAL
        <NodeType.NORMAL: 0>
        >>> NodeType.NORMAL.value
        0
        >>> int(NodeType.WALL_BOUNDARY)
        6
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
