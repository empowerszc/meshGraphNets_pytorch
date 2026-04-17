"""
Rollout - 长时序自回归推理脚本

本脚本实现了 MeshGraphNet 模型的长时序推理 (rollout) 功能，
用于评估模型在长期预测中的稳定性和误差累积情况。

什么是 Rollout?
──────────────────────────────────────────────────────────────
Rollout 是指从初始条件开始，反复使用模型预测下一时刻状态，
并将预测结果作为下一步的输入，如此循环进行长期预测。

与普通推理的区别:
- 普通推理：单步预测，输入是真值
- Rollout: 多步自回归预测，输入是上一步的预测值

为什么需要 Rollout 评估？
──────────────────────────────────────────────────────────────
1. 模拟真实使用场景：实际使用时就是自回归推理
2. 检测误差累积：单步准确 ≠ 长期稳定
3. 评估数值稳定性：误差是否随时间指数增长

使用方法:
    # 单轨迹 rollout
    python rollout.py

    # 多轨迹 rollout
    python rollout.py --rollout_num 5

    # 指定 GPU
    python rollout.py --gpu 1

    # 指定模型
    python rollout.py --model_dir checkpoints/best_model.pth
"""

import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

from dataset import FpcDataset
from model.simulator import Simulator
from utils.utils import NodeType


def rollout_error(predicteds, targets):
    """
    计算 rollout 累积误差

    这是评估长期预测稳定性的关键指标。

    计算方法:
    1. 计算每步的平方误差
    2. 计算累积平均 MSE
    3. 开方得到累积 RMSE

    Args:
        predicteds: 预测速度序列 [num_steps, N, 2]
        targets: 目标速度序列 [num_steps, N, 2]

    Returns:
        累积 RMSE 曲线 [num_steps]

    误差累积公式:
    ┌─────────────────────────────────────────────────────────┐
    │ 第 t 步的累积 RMSE:                                      │
    │                                                         │
    │ RMSE_cumulative(t) = sqrt(Σᵢ₌₁ᵗ MSE(i) / t)           │
    │                                                         │
    │ 其中：MSE(i) = mean((predicted_i - target_i)²)         │
    └─────────────────────────────────────────────────────────┘

    输出示例:
        testing rmse  @ step 0 loss: 1.23e-03
        testing rmse  @ step 50 loss: 2.45e-03
        testing rmse  @ step 100 loss: 4.67e-03
        ...
    """
    number_len = targets.shape[0]  # 时间步数

    # 重塑为 [num_steps, N*2]，方便计算
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)

    # 计算累积平均 MSE，然后开方得到 RMSE
    # np.cumsum(np.mean(...)) / np.arange(1, number_len+1)
    loss = np.sqrt(
        np.cumsum(np.mean(squared_diff, axis=1), axis=0) /
        np.arange(1, number_len + 1)
    )

    # 打印关键步的误差
    for show_step in range(0, 1000000, 50):
        if show_step < number_len:
            print('testing rmse  @ step %d loss: %.2e' % (show_step, loss[show_step]))
        else:
            break
    return loss


@torch.no_grad()
def rollout(model, dataset, rollout_index=1):
    """
    执行单条轨迹的 rollout 推理

    Args:
        model: Simulator 模型 (已加载训练好的权重)
        dataset: 测试数据集
        rollout_index: 要推理的轨迹索引
                      - 0: 第一条轨迹
                      - 1: 第二条轨迹
                      - ...

    Returns:
        [predicteds, targets]: 预测序列和目标序列

    Rollout 流程:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 初始化：predicted_velocity = None                    │
    │                                                         │
    │ 2. for i in range(轨迹长度):                            │
    │    ├─ 加载第 i 步的图数据                                │
    │    ├─ 如果 i > 0: 用上一帧预测替换当前输入速度          │
    │    ├─ 记录边界条件掩码 (需要固定的节点)                 │
    │    ├─ 模型前向传播 → predicted_velocity                 │
    │    ├─ 边界条件约束：强制重置边界/入口速度               │
    │    └─ 保存预测和真值                                    │
    │                                                         │
    │ 3. 保存结果到 result/result{rollout_index}.pkl          │
    └─────────────────────────────────────────────────────────┘

    关键点:
    1. 自回归：用预测值作为下一步输入
    2. 边界约束：边界节点速度强制重置为真值
    3. 无梯度：torch.no_grad() 节省内存
    """
    num_sampes_per_tra = dataset.num_sampes_per_tra  # 轨迹长度

    predicted_velocity = None  # 上一帧的预测速度
    mask = None                 # 边界条件掩码
    predicteds = []             # 预测序列
    targets = []                # 目标序列

    for i in range(num_sampes_per_tra):
        # ========== 1. 加载当前帧数据 ==========
        # 计算全局索引：第 rollout_index 条轨迹的第 i 步
        index = rollout_index * num_sampes_per_tra + i
        graph = dataset[index]

        # 数据预处理：生成边特征
        graph = transformer(graph)
        graph = graph.cuda()

        # ========== 2. 获取边界条件掩码 ==========
        # 只在第一帧计算掩码 (网格拓扑不变，掩码也不变)
        if mask is None:
            node_type = graph.x[:, 0]
            # 需要预测的节点：NORMAL + OUTFLOW
            mask = torch.logical_or(
                node_type == NodeType.NORMAL,
                node_type == NodeType.OUTFLOW
            )
            # 取反：需要固定的节点 (边界、入口、障碍物等)
            mask = torch.logical_not(mask)

        # ========== 3. 【关键】自回归输入 ==========
        # 如果不是第一步，用上一帧的预测替换当前输入速度
        if predicted_velocity is not None:
            graph.x[:, 1:3] = predicted_velocity.detach()

        # ========== 4. 获取真值 (用于边界约束和评估) ==========
        next_v = graph.y  # 下一时刻的真实速度

        # ========== 5. 模型预测 ==========
        with torch.no_grad():
            predicted_velocity = model(graph, velocity_sequence_noise=None)

        # ========== 6. 【关键】边界条件约束 ==========
        # 强制将边界节点的速度重置为真值
        # 这是物理模拟的基本要求：边界条件必须满足
        predicted_velocity[mask] = next_v[mask]

        # ========== 7. 保存结果 ==========
        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

    # 获取网格坐标 (用于可视化)
    crds = graph.pos.cpu().numpy()

    # 堆叠为 [num_steps, N, 2]
    result = [np.stack(predicteds), np.stack(targets)]

    # 保存到 pickle 文件
    os.makedirs('result', exist_ok=True)
    with open('result/result' + str(rollout_index) + '.pkl', 'wb') as f:
        pickle.dump([result, crds], f)

    return result


if __name__ == '__main__':
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')

    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU 编号：0 或 1")

    parser.add_argument("--model_dir",
                        type=str,
                        default='checkpoints/best_model.pth',
                        help="模型 checkpoint 路径")

    parser.add_argument("--test_split",
                        type=str,
                        default='test',
                        help="测试集划分：test/valid")

    parser.add_argument("--rollout_num",
                        type=int,
                        default=1,
                        help="要执行的 rollout 轨迹数量")

    args = parser.parse_args()

    # ==================== 加载模型 ====================
    torch.cuda.set_device(args.gpu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 创建模型
    simulator = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device=device
    )

    # 加载训练好的权重
    state_dict = torch.load(args.model_dir, weights_only=False)
    simulator.load_state_dict(state_dict['model_state_dict'])
    simulator.eval()  # 设置为评估模式

    # ==================== 准备数据集 ====================
    dataset_dir = "data"
    dataset = FpcDataset(dataset_dir, split=args.test_split)

    # 数据预处理变换 (与训练时一致)
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # ==================== 执行 Rollout ====================
    for i in range(args.rollout_num):
        print(f"\n{'='*60}")
        print(f"Starting rollout #{i}...")
        print('='*60)

        result = rollout(simulator, dataset, rollout_index=i)

        print('------------------------------------------------------------------')
        print("Rollout 误差分析:")
        rollout_error(result[0], result[1])
