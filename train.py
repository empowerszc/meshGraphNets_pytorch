"""
Train - 单 GPU 训练脚本

本脚本实现了 MeshGraphNet 模型的完整训练流程，包括：
1. 数据加载与预处理
2. 模型初始化
3. 训练循环 (带进度条)
4. 验证评估
5. TensorBoard 日志记录
6. 最优模型保存

训练流程概览:
┌─────────────────────────────────────────────────────────────┐
│ 1. 加载数据集 (FpcDataset)                                  │
│ 2. 数据预处理 (FaceToEdge → Cartesian → Distance)           │
│ 3. 初始化模型 (Simulator) 和优化器 (Adam)                   │
│ 4. for epoch in range(num_epochs):                          │
│    ├─ train_one_epoch()                                     │
│    │   ├─ 注入噪声                                          │
│    │   ├─ 预测加速度                                        │
│    │   ├─ 计算 Loss (仅 NORMAL + OUTFLOW)                   │
│    │   └─ 反向传播更新                                      │
│    └─ evaluate()                                            │
│        └─ 验证集 RMSE 评估                                   │
│ 5. 保存最优模型 checkpoint                                  │
└─────────────────────────────────────────────────────────────┘

使用方法:
    python train.py

多 GPU 训练请使用 train_ddp.py
"""

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
from dataset import FpcDataset
from model.simulator import Simulator
from utils.noise import get_velocity_noise
from utils.utils import NodeType
import os
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

# ==================== 配置参数 ====================
dataset_dir = "data"           # 数据集目录
batch_size = 20                # 批大小
noise_std = 2e-2               # 噪声标准差 (训练技巧)
num_epochs = 100               # 训练轮数

# 设备选择：优先使用 GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 保存目录
checkpoint_dir = "checkpoints"  # 模型 checkpoint 保存目录
log_dir = "runs"                # TensorBoard 日志目录
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ==================== 模型与优化器初始化 ====================
# 创建 Simulator 模型
# 参数与论文一致：15 层消息传递，节点输入 11 维，边输入 3 维
simulator = Simulator(
    message_passing_num=15,
    node_input_size=11,
    edge_input_size=3,
    device=device
)

# Adam 优化器
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')

# TensorBoard writer (用于可视化训练曲线)
writer = SummaryWriter(log_dir=log_dir)

# ==================== 数据预处理 ====================
# PyG 变换组合：将网格面转换为边特征
transformer = T.Compose([
    T.FaceToEdge(),           # 从三角面生成边 (face → edge_index)
    T.Cartesian(norm=False),  # 添加相对坐标作为边特征 (dx, dy)
    T.Distance(norm=False)    # 添加距离作为边特征 (||dx||)
])
# 最终边特征维度：3 = 2(相对坐标) + 1(距离)


def train_one_epoch(model: Simulator, dataloader, optimizer, transformer, device, noise_std):
    """
    训练一个 epoch

    Args:
        model: Simulator 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        transformer: 数据预处理变换
        device: 运行设备
        noise_std: 噪声标准差

    Returns:
        平均训练 loss

    训练步骤:
    1. 加载 batch 数据
    2. 应用 transformer 生成边特征
    3. 注入速度噪声 (提高鲁棒性)
    4. 前向传播：预测归一化加速度
    5. 计算 Loss：仅在 NORMAL 和 OUTFLOW 节点上计算 MSE
    6. 反向传播更新参数
    """
    model.train()  # 设置为训练模式
    total_loss = 0.0
    num_batches = 0

    for graph in tqdm.tqdm(dataloader, desc="Training"):
        # 数据预处理：生成边特征
        graph = transformer(graph)
        graph = graph.to(device)

        # 提取节点类型用于 mask 计算
        node_type = graph.x[:, 0]  # [N]

        # 生成速度噪声 (仅在 NORMAL 节点上加噪)
        velocity_sequence_noise = get_velocity_noise(
            graph, noise_std=noise_std, device=device
        )

        # 前向传播：返回 (预测加速度，目标加速度)
        # 两者都是归一化后的值
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)

        # 创建 Loss 计算掩码：仅预测流体区域和出口
        # 边界、障碍物等节点不参与 Loss 计算
        mask = torch.logical_or(
            node_type == NodeType.NORMAL,
            node_type == NodeType.OUTFLOW
        )

        # 计算 MSE Loss
        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = torch.mean(errors)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model: Simulator, dataloader, transformer, device):
    """
    在验证集上评估模型

    Args:
        model: Simulator 模型
        dataloader: 验证数据加载器
        transformer: 数据预处理变换
        device: 运行设备

    Returns:
        平均验证 Loss (RMSE)

    评估步骤:
    1. 无梯度模式 (torch.no_grad)
    2. 推理模式前向传播 (直接输出速度预测)
    3. 计算 RMSE Loss
    """
    model.eval()  # 设置为评估模式
    losses = []

    with torch.no_grad():  # 不计算梯度，节省内存
        for graph in dataloader:
            # 数据预处理
            graph = transformer(graph)
            graph = graph.to(device)

            node_type = graph.x[:, 0]

            # 推理模式：velocity_sequence_noise=None
            # 模型直接返回下一时刻速度预测
            predicted_velocity = model(graph, None)

            # 创建 Loss 计算掩码 (与训练一致)
            mask = torch.logical_or(
                node_type == NodeType.NORMAL,
                node_type == NodeType.OUTFLOW
            )

            # 计算 RMSE Loss
            errors = ((predicted_velocity - graph.y) ** 2)[mask]
            loss = torch.sqrt(torch.mean(errors))
            losses.append(loss.item())

    return np.mean(losses)


if __name__ == '__main__':
    # ==================== 加载数据集 ====================
    train_dataset = FpcDataset(data_root=dataset_dir, split='train')
    valid_dataset = FpcDataset(data_root=dataset_dir, split='valid')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 训练集打乱
        num_workers=2
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,     # 验证集不打乱
        num_workers=2
    )

    # 模型移入设备
    simulator.to(device)

    # ==================== 训练循环 ====================
    best_valid_loss = float('inf')  # 最优验证 loss
    best_epoch = -1                  # 最优模型所在 epoch

    for epoch in range(1, num_epochs + 1):
        # 训练一个 epoch
        train_loss = train_one_epoch(
            simulator, train_loader, optimizer,
            transformer, device, noise_std
        )

        # 验证评估
        valid_loss = evaluate(simulator, valid_loader, transformer, device)

        # 打印进度
        print(f"Epoch {epoch}/{num_epochs} "
              f"Train Loss: {train_loss:.2e} "
              f"Valid Loss: {valid_loss:.2e}")

        # TensorBoard 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': simulator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} "
                  f"with valid loss {valid_loss:.2e}")

    # 训练结束
    writer.close()
    print(f"\nTraining finished. "
          f"Best model at epoch {best_epoch} "
          f"with validation loss {best_valid_loss:.2e}")
