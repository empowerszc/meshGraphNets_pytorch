#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo Inference - MeshGraphNet 模型推理演示脚本

本脚本用于在准备好环境后，快速运行模型推理并查看效果。
支持多种运行模式，包括随机初始化参数推理和加载训练好的模型推理。

功能特点:
    1. ✅ 自动检测并适配 CPU/GPU 环境
    2. ✅ 支持随机初始化参数推理 (无需训练即可测试)
    3. ✅ 支持加载训练好的 checkpoint
    4. ✅ 生成简单的测试数据 (正弦波速度场)
    5. ✅ 输出可视化结果 (速度场变化图)
    6. ✅ 支持单步预测和多步 rollout 推理
    7. ✅ 详细的进度和统计信息输出

使用方法:
    # 模式 1: 随机初始化参数推理 (快速测试)
    python demo_inference.py --mode random

    # 模式 2: 加载训练好的模型推理
    python demo_inference.py --mode checkpoint --checkpoint_path checkpoints/best_model.pth

    # 模式 3: 多步 rollout 推理
    python demo_inference.py --mode random --rollout_steps 50

    # 模式 4: 使用 CPU (强制)
    python demo_inference.py --mode random --device cpu

    # 模式 5: 完整演示 (所有功能)
    python demo_inference.py --mode random --full_demo
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 导入项目模块
from model.simulator import Simulator
from utils.utils import NodeType


def print_header(text):
    """打印带分隔线的标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    """打印小节标题"""
    print(f"\n--- {text} ---")


def detect_device(device_arg):
    """
    自动检测并选择合适的设备

    Args:
        device_arg: 命令行参数 ('cuda', 'cpu', 或 'auto')

    Returns:
        device: torch.device 对象
        device_name: 设备名称字符串
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
        else:
            print("⚠️  警告：CUDA 不可用，回退到 CPU")
            device = torch.device('cpu')
            device_name = "CPU"
    else:
        device = torch.device('cpu')
        device_name = "CPU"

    return device, device_name


def generate_test_mesh(num_nodes_x=20, num_nodes_y=15, domain_size=(2.0, 1.5)):
    """
    生成简单的矩形网格测试数据

    Args:
        num_nodes_x: X 方向节点数
        num_nodes_y: Y 方向节点数
        domain_size: 计算域大小 (width, height)

    Returns:
        pos: 节点位置 [N, 2]
        cells: 三角单元 [num_faces, 3]
        node_type: 节点类型 [N, 1]
    """
    width, height = domain_size

    # 生成网格节点
    x = np.linspace(0, width, num_nodes_x)
    y = np.linspace(0, height, num_nodes_y)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # [N, 2]

    # 生成三角单元 (Delaunay 三角化)
    triang = tri.Triangulation(pos[:, 0], pos[:, 1])
    cells = triang.triangles.astype(np.int64)  # [num_faces, 3]

    # 生成节点类型
    N = pos.shape[0]
    node_type = np.full((N, 1), NodeType.NORMAL, dtype=np.float32)

    # 标记边界节点
    tol = 1e-6
    for i in range(N):
        # 底部和顶部边界 -> WALL
        if pos[i, 1] < tol or pos[i, 1] > height - tol:
            node_type[i] = NodeType.WALL_BOUNDARY
        # 左侧边界 -> INFLOW
        elif pos[i, 0] < tol:
            node_type[i] = NodeType.INFLOW
        # 右侧边界 -> OUTFLOW
        elif pos[i, 0] > width - tol:
            node_type[i] = NodeType.OUTFLOW

    return pos, cells, node_type


def generate_initial_velocity(pos, pattern='sine'):
    """
    生成初始速度场

    Args:
        pos: 节点位置 [N, 2]
        pattern: 速度场模式 ('sine', 'uniform', 'vortex')

    Returns:
        velocity: 初始速度 [N, 2]
    """
    N = pos.shape[0]
    velocity = np.zeros((N, 2), dtype=np.float32)

    if pattern == 'sine':
        # 正弦波速度场 (模拟卡门涡街)
        x, y = pos[:, 0], pos[:, 1]
        # 主流速度 + 横向扰动
        velocity[:, 0] = 1.0 + 0.5 * np.sin(2 * np.pi * y / np.max(y))
        velocity[:, 1] = 0.3 * np.sin(2 * np.pi * x / np.max(x))

    elif pattern == 'uniform':
        # 均匀来流
        velocity[:, 0] = 1.0
        velocity[:, 1] = 0.0

    elif pattern == 'vortex':
        # 涡旋场
        x, y = pos[:, 0] - np.max(pos[:, 0]) / 2, pos[:, 1] - np.max(pos[:, 1]) / 2
        r = np.sqrt(x**2 + y**2) + 1e-6
        velocity[:, 0] = -y / r
        velocity[:, 1] = x / r

    return velocity


def create_test_graph(pos, cells, node_type, velocity):
    """
    创建 PyG Data 对象

    Args:
        pos: 节点位置
        cells: 三角单元
        node_type: 节点类型 [N, 1]
        velocity: 速度场 [N, 2]

    Returns:
        graph: PyG Data 对象
    """
    # node_type 需要是整数类型用于 one_hot 编码
    node_type_int = node_type.astype(np.int64)  # [N, 1]
    x = np.concatenate([node_type_int, velocity], axis=-1)  # [N, 1+2=3]
    x = torch.as_tensor(x, dtype=torch.float32)
    pos_tensor = torch.as_tensor(pos, dtype=torch.float32)
    face_tensor = torch.as_tensor(cells.T, dtype=torch.int64)  # [3, num_faces]

    # y 占位符 (推理时不需要，但模型可能需要)
    y = torch.zeros_like(x[:, :2], dtype=torch.float32)  # [N, 2]

    graph = Data(x=x, pos=pos_tensor, face=face_tensor, y=y)
    return graph


def apply_transform(graph):
    """应用 PyG 变换，生成边特征"""
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])
    return transformer(graph)


def visualize_results(predicteds, targets, pos, cells, save_path=None):
    """
    可视化推理结果

    Args:
        predicteds: 预测速度序列 [steps, N, 2]
        targets: 目标速度序列 [steps, N, 2]
        pos: 节点位置
        cells: 三角单元
        save_path: 保存路径 (可选)
    """
    num_steps = len(predicteds)
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=cells)

    # 计算速度范数
    target_speed = np.linalg.norm(targets, axis=-1)
    predicted_speed = np.linalg.norm(predicteds, axis=-1)
    v_max = max(target_speed.max(), predicted_speed.max())
    v_min = min(target_speed.min(), predicted_speed.min())

    # 创建图表
    num_cols = min(num_steps, 4)
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))

    # 处理 axes 维度
    if num_cols == 1:
        axes = np.array([[axes[0], axes[1]]]).T  # reshape to (2, 1)
    elif num_cols == 2:
        axes = axes.reshape(2, 2)
    # num_cols >= 3 时 axes 已经是 (2, num_cols)

    steps_to_show = list(range(num_cols))

    for j, step in enumerate(steps_to_show):
        # 目标
        ax = axes[0, j]
        ax.triplot(triang, 'k-', alpha=0.3, lw=0.5)
        cf = ax.tripcolor(triang, target_speed[step], vmin=v_min, vmax=v_max)
        ax.set_title(f'Target (step {step})')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(cf, ax=ax)

        # 预测
        ax = axes[1, j]
        ax.triplot(triang, 'k-', alpha=0.3, lw=0.5)
        cf = ax.tripcolor(triang, predicted_speed[step], vmin=v_min, vmax=v_max)
        ax.set_title(f'Prediction (step {step})')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(cf, ax=ax)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到：{save_path}")
    else:
        plt.show()

    plt.close()


def inference_random(device, args):
    """
    使用随机初始化的模型进行推理

    Args:
        device: torch.device
        args: 命令行参数
    """
    print_header("随机初始化模型推理")

    print_section("1. 生成测试网格")
    pos, cells, node_type = generate_test_mesh(
        num_nodes_x=args.mesh_nx,
        num_nodes_y=args.mesh_ny
    )
    print(f"网格规模：{pos.shape[0]} 节点，{cells.shape[0]} 三角单元")
    print(f"节点类型分布:")
    for nt in NodeType:
        if nt != NodeType.SIZE:
            count = np.sum(node_type == int(nt))
            print(f"  - {nt.name}: {count} 节点")

    print_section("2. 生成初始速度场")
    velocity = generate_initial_velocity(pos, pattern=args.velocity_pattern)
    print(f"初始速度场模式：{args.velocity_pattern}")
    print(f"速度范围：[{velocity.min():.4f}, {velocity.max():.4f}]")

    print_section("3. 创建图数据")
    graph = create_test_graph(pos, cells, node_type, velocity)
    graph = apply_transform(graph)
    graph = graph.to(device)
    print(f"节点特征维度：{graph.x.shape[1]}")
    print(f"边特征维度：{graph.edge_attr.shape[1]}")
    print(f"边数量：{graph.edge_index.shape[1]}")

    print_section("4. 初始化模型")
    model = Simulator(
        message_passing_num=args.message_passing_num,
        node_input_size=11,
        edge_input_size=3,
        device=device
    )
    model.eval()
    print(f"消息传递层数：{args.message_passing_num}")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    print_section("5. 执行推理")

    if args.rollout_steps > 1:
        # 多步 rollout 推理
        print(f"Rollout 推理：{args.rollout_steps} 步")
        predicteds = []
        targets = []

        current_velocity = velocity.copy()

        for step in range(args.rollout_steps):
            # 更新输入速度
            graph.x[:, 1:3] = torch.as_tensor(current_velocity, dtype=torch.float32).to(device)

            # 模型预测
            with torch.no_grad():
                predicted_velocity = model(graph, None)

            predicted_velocity = predicted_velocity.cpu().numpy()

            # 边界条件约束
            boundary_mask = (node_type != int(NodeType.NORMAL)).flatten()
            current_velocity[boundary_mask] = velocity[boundary_mask]  # 保持初始边界
            current_velocity[~boundary_mask] = predicted_velocity[~boundary_mask]

            predicteds.append(current_velocity.copy())
            targets.append(velocity.copy())  # 简单起见，target 保持初始值

            if (step + 1) % 10 == 0:
                print(f"  步骤 {step + 1}/{args.rollout_steps} 完成")

    else:
        # 单步推理
        print("单步推理")
        with torch.no_grad():
            predicted_velocity = model(graph, None)
        predicteds = [predicted_velocity.cpu().numpy()]
        targets = [velocity.copy()]

    print_section("6. 结果统计")
    print(f"预测速度范围：[{np.array(predicteds).min():.4f}, {np.array(predicteds).max():.4f}]")
    print(f"目标速度范围：[{np.array(targets).min():.4f}, {np.array(targets).max():.4f}]")

    # 计算误差
    errors = np.sqrt(np.mean((np.array(predicteds) - np.array(targets)) ** 2, axis=-1))
    print(f"平均绝对误差：{errors.mean():.4f}")
    print(f"最大绝对误差：{errors.max():.4f}")

    print_section("7. 可视化")
    vis_path = "outputs/demo_inference_result.png" if args.save_vis else None
    visualize_results(np.array(predicteds), np.array(targets), pos, cells, save_path=vis_path)

    print_header("推理完成!")


def inference_checkpoint(device, args):
    """
    使用训练好的 checkpoint 进行推理

    Args:
        device: torch.device
        args: 命令行参数
    """
    print_header("加载训练好的模型进行推理")

    # 检查 checkpoint 文件
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ 错误：找不到 checkpoint 文件：{args.checkpoint_path}")
        print("请先训练模型或指定正确的路径")
        return

    print_section("1. 加载模型")
    model = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device=device
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"已加载 checkpoint: {args.checkpoint_path}")
    print(f"训练轮数：{checkpoint.get('epoch', 'N/A')}")
    print(f"验证 Loss: {checkpoint.get('valid_loss', 'N/A')}")

    print_section("2. 准备测试数据")
    # 如果有真实数据集，加载它
    # 这里使用简化版本，生成测试数据
    pos, cells, node_type = generate_test_mesh(
        num_nodes_x=args.mesh_nx,
        num_nodes_y=args.mesh_ny
    )
    velocity = generate_initial_velocity(pos, pattern=args.velocity_pattern)
    print(f"测试网格：{pos.shape[0]} 节点")

    print_section("3. 创建图数据")
    graph = create_test_graph(pos, cells, node_type, velocity)
    graph = apply_transform(graph)
    graph = graph.to(device)

    print_section("4. 执行推理")
    predicteds = []
    targets = []
    current_velocity = velocity.copy()

    num_steps = args.rollout_steps
    for step in range(num_steps):
        graph.x[:, 1:3] = torch.as_tensor(current_velocity, dtype=torch.float32).to(device)

        with torch.no_grad():
            predicted_velocity = model(graph, None)

        predicted_velocity = predicted_velocity.cpu().numpy()

        boundary_mask = (node_type != int(NodeType.NORMAL)).flatten()
        current_velocity[~boundary_mask] = predicted_velocity[~boundary_mask]

        predicteds.append(current_velocity.copy())
        targets.append(velocity.copy())

        if (step + 1) % 10 == 0:
            print(f"  步骤 {step + 1}/{num_steps} 完成")

    print_section("5. 结果可视化")
    vis_path = "outputs/checkpoint_inference_result.png" if args.save_vis else None
    visualize_results(np.array(predicteds), np.array(targets), pos, cells, save_path=vis_path)

    print_header("推理完成!")


def main():
    parser = argparse.ArgumentParser(
        description='MeshGraphNet 模型推理演示脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo_inference.py --mode random                    # 随机初始化模型推理
  python demo_inference.py --mode random --rollout-steps 50  # 50 步 rollout
  python demo_inference.py --mode checkpoint -c model.pth   # 加载 checkpoint
  python demo_inference.py --mode random --device cpu       # 使用 CPU
  python demo_inference.py --mode random --full-demo        # 完整演示
        """
    )

    # 基本参数
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'checkpoint'],
                        help='推理模式：random(随机初始化) 或 checkpoint(加载训练好的模型)')
    parser.add_argument('--checkpoint_path', '-c', type=str,
                        help='checkpoint 文件路径 (mode=checkpoint 时需要)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='设备选择：auto(自动), cuda, cpu')

    # 模型参数
    parser.add_argument('--message_passing_num', type=int, default=15,
                        help='消息传递层数 (默认 15)')

    # 网格参数
    parser.add_argument('--mesh_nx', type=int, default=20,
                        help='X 方向网格节点数')
    parser.add_argument('--mesh_ny', type=int, default=15,
                        help='Y 方向网格节点数')

    # 速度场参数
    parser.add_argument('--velocity_pattern', type=str, default='sine',
                        choices=['sine', 'uniform', 'vortex'],
                        help='初始速度场模式')

    # 推理参数
    parser.add_argument('--rollout_steps', type=int, default=1,
                        help='Rollout 推理步数')

    # 输出参数
    parser.add_argument('--save_vis', action='store_true',
                        help='是否保存可视化结果到文件')
    parser.add_argument('--full_demo', action='store_true',
                        help='完整演示模式 (自动保存可视化结果)')

    args = parser.parse_args()

    # full_demo 模式自动保存可视化
    if args.full_demo:
        args.save_vis = True

    # 创建输出目录
    if args.save_vis:
        os.makedirs('outputs', exist_ok=True)

    # 打印配置
    print("\n" + "=" * 70)
    print("  MeshGraphNet 推理配置")
    print("=" * 70)
    print(f"推理模式：{args.mode}")
    print(f"设备：{args.device}")
    print(f"网格规模：{args.mesh_nx} x {args.mesh_ny}")
    print(f"速度场模式：{args.velocity_pattern}")
    print(f"Rollout 步数：{args.rollout_steps}")
    print(f"消息传递层数：{args.message_passing_num}")
    print(f"保存可视化：{'是' if args.save_vis else '否'}")
    print("=" * 70)

    # 检测设备
    device, device_name = detect_device(args.device)
    print(f"使用设备：{device_name}")

    # 选择推理模式
    if args.mode == 'random':
        inference_random(device, args)
    elif args.mode == 'checkpoint':
        if not args.checkpoint_path:
            print("❌ 错误：checkpoint 模式需要指定 --checkpoint_path")
            sys.exit(1)
        inference_checkpoint(device, args)


if __name__ == '__main__':
    main()
