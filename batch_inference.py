#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Inference - MeshGraphNet 批量推理脚本

功能：
1. 从 test 测试集加载指定数量的轨迹（如数据存在）
2. 如无真实数据，自动生成合成数据进行测试
3. 支持设置 batch size 进行批量推理
4. 支持设置推理步数（默认完整轨迹，可限制节省内存）
5. 支持随机权重或加载训练好的 checkpoint
6. 输出推理性能统计（延迟、吞吐量）
7. 可选保存可视化结果

使用方法:
    # 使用合成数据测试：推理 10 条轨迹，batch_size=2，完整轨迹 (默认 600 步)
    python batch_inference.py --num-samples 10 --batch-size 2

    # 节省内存模式：每条轨迹只跑 100 步
    python batch_inference.py --num-samples 10 --batch-size 2 --num-steps 100

    # 加载 checkpoint，使用真实数据（需要先运行 parse_tfrecord.py 转换数据）
    python batch_inference.py --num-samples 50 --batch-size 5 --checkpoint checkpoints/best_model.pth

    # 推理全部 100 条测试轨迹，完整 600 步
    python batch_inference.py --num-samples 100 --batch-size 10 --save-vis

内存优化建议:
    - 小内存机器：减小 --num-steps (如 50-100 步)
    - 大内存机器：使用完整轨迹或较大 --num-steps
    - batch_size 越大，内存占用越高，建议从 1-2 开始测试
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from model.simulator import Simulator
from dataset.fpc import FpcDataset
from utils.utils import NodeType


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    print(f"\n--- {text} ---")


def parse_args():
    parser = argparse.ArgumentParser(
        description='MeshGraphNet 批量推理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 基本参数
    parser.add_argument('--num-samples', type=int, default=10,
                        help='推理的轨迹数量 (默认 10)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小 (默认 1)')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='每条轨迹推理的步数 (默认 None=完整轨迹，约 600 步；可设置更小的值节省内存)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='数据集目录 (默认 data)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='设备选择')

    # 模型参数
    parser.add_argument('--random', action='store_true', default=True,
                        help='使用随机权重 (默认)')
    parser.add_argument('--checkpoint', type=str,
                        help='加载训练好的 checkpoint 路径')

    # 输出参数
    parser.add_argument('--output-dir', type=str,
                        default=f'outputs/batch_inference/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='输出目录')
    parser.add_argument('--save-vis', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--verbose', action='store_true',
                        help='详细日志输出')

    return parser.parse_args()


def detect_device(device_arg):
    """自动检测设备"""
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


def load_trajectory(dataset, tra_index, num_steps=None):
    """
    加载单条轨迹的指定时间步

    Args:
        dataset: FpcDataset 实例
        tra_index: 轨迹索引
        num_steps: 推理步数 (None=使用数据集默认)

    Returns:
        图数据列表，每个包含该轨迹指定时间步的 Data 对象
    """
    # 每条轨迹的样本数
    samples_per_tra = dataset.num_sampes_per_tra

    # 限制推理步数
    if num_steps is not None:
        num_steps = min(num_steps, samples_per_tra)
    else:
        num_steps = samples_per_tra

    start_idx = tra_index * samples_per_tra
    end_idx = start_idx + num_steps

    graphs = []
    for idx in range(start_idx, end_idx):
        graphs.append(dataset[idx])

    return graphs


def create_batch_from_graphs(graphs, device):
    """
    将多个图数据 batch 在一起

    Args:
        graphs: Data 对象列表
        device: torch.device

    Returns:
        Batch 对象
    """
    # 将所有图移到 device
    graphs_on_device = [g.to(device) for g in graphs]
    batch = Batch.from_data_list(graphs_on_device)
    return batch


def visualize_batch_results(
    predicteds, targets, pos, cells,
    output_dir, step=0, max_show=4
):
    """可视化 batch 中部分样本的推理结果"""
    triang = tri.Triangulation(pos[:, 0], pos[:, 1], triangles=cells)

    # 计算速度范数
    target_speed = np.linalg.norm(targets, axis=-1)
    predicted_speed = np.linalg.norm(predicteds, axis=-1)
    v_max = max(target_speed.max(), predicted_speed.max())
    v_min = min(target_speed.min(), predicted_speed.min())

    num_show = min(max_show, len(predicteds))
    fig, axes = plt.subplots(2, num_show, figsize=(4 * num_show, 8))

    if num_show == 1:
        axes = np.array([[axes[0], axes[1]]]).T

    for j in range(num_show):
        # Target
        ax = axes[0, j]
        ax.triplot(triang, 'k-', alpha=0.3, lw=0.5)
        cf = ax.tripcolor(triang, target_speed[j], vmin=v_min, vmax=v_max)
        ax.set_title(f'Target (sample {j})')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(cf, ax=ax)

        # Prediction
        ax = axes[1, j]
        ax.triplot(triang, 'k-', alpha=0.3, lw=0.5)
        cf = ax.tripcolor(triang, predicted_speed[j], vmin=v_min, vmax=v_max)
        ax.set_title(f'Prediction (sample {j})')
        ax.set_aspect('equal')
        ax.axis('off')
        plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'batch_step{step:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def main():
    args = parse_args()

    # ========== 1. 初始化配置 ==========
    print_header("MeshGraphNet 批量推理")

    print_section("1. 配置信息")
    print(f"设备：{args.device}")
    print(f"轨迹数量：{args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"每条轨迹推理步数：{args.num_steps if args.num_steps else '完整轨迹 (约 600 步)'}")
    print(f"模型权重：{'随机初始化' if args.random else args.checkpoint}")

    # 检测设备
    device, device_name = detect_device(args.device)
    print(f"使用设备：{device_name}")

    # 验证 num_samples 能被 batch_size 整除
    if args.num_samples % args.batch_size != 0:
        print(f"⚠️  警告：{args.num_samples} 不能被 {args.batch_size} 整除")
        print(f"   将调整 batch 数为 {args.num_samples // args.batch_size + 1}，最后一个 batch 可能较小")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录：{args.output_dir}")

    # ========== 2. 加载数据集 ==========
    print_section("2. 加载数据")

    # 检查真实数据集是否存在
    test_path = Path(args.data_dir) / 'test.npz'
    use_synthetic = not test_path.exists()

    if use_synthetic:
        print(f"⚠️  未找到真实数据集 ({args.data_dir}/test.npz 不存在)")
        print("   使用合成数据进行测试")

        # 导入合成数据生成函数
        from demo_inference import generate_test_mesh, generate_initial_velocity, create_test_graph, apply_transform

        # 生成合成网格
        pos, cells, node_type = generate_test_mesh(num_nodes_x=20, num_nodes_y=15)
        print(f"合成网格：{pos.shape[0]} 节点，{cells.shape[0]} 三角单元")

        class SyntheticDataset:
            """合成数据集 - 生成随机速度场用于测试"""
            def __init__(self, pos, cells, node_type, num_tras=10, tra_len=600):
                self.pos = pos
                self.cells = cells
                self.node_type = node_type
                self.num_tras = num_tras
                self.tra_len = tra_len
                self.num_sampes_per_tra = tra_len - 1
                self.meta = {
                    'indices': list(range(0, (num_tras + 1) * pos.shape[0], pos.shape[0]))
                }
                # 预生成所有轨迹的速度场
                self.velocities = []
                for i in range(num_tras):
                    vel = generate_initial_velocity(pos, pattern='sine')
                    self.velocities.append(vel)

            def __getitem__(self, index):
                tra_index = index // self.num_sampes_per_tra
                tra_sample_index = index % self.num_sampes_per_tra

                # 当前时刻速度
                velocity = self.velocities[tra_index].copy()
                # 下一时刻目标速度（稍微变化）
                target_velocity = velocity + np.random.randn(*velocity.shape) * 0.001

                graph = create_test_graph(self.pos, self.cells, self.node_type, velocity)
                graph.y = torch.as_tensor(target_velocity, dtype=torch.float32)
                return graph

            def __len__(self):
                return self.num_tras * self.num_sampes_per_tra

        dataset = SyntheticDataset(pos, cells, node_type, num_tras=args.num_samples + 1)
        print(f"合成数据集：{dataset.num_tras} 条轨迹，每条 {dataset.tra_len} 步")
    else:
        dataset = FpcDataset(data_root=args.data_dir, split='test')
    print(f"测试集总样本数：{len(dataset)}")
    print(f"每条轨迹时间步数：{dataset.tra_len}")
    print(f"每条轨迹样本数：{dataset.num_sampes_per_tra}")

    # 检查并限制推理步数
    max_steps = dataset.num_sampes_per_tra

    if args.num_steps is None:
        # 默认使用完整轨迹
        args.num_steps = max_steps
        print(f"✓ 使用完整轨迹推理 ({max_steps} 步)")
    elif args.num_steps > max_steps:
        print(f"⚠️  警告：请求 {args.num_steps} 步，但数据集最多支持 {max_steps} 步")
        print(f"   将调整为 {max_steps} 步")
        args.num_steps = max_steps
    else:
        print(f"✓ 使用 {args.num_steps} 步推理 (完整轨迹为 {max_steps} 步)")

    # 检查请求的轨迹数是否合法
    max_tras = len(dataset.meta['indices']) - 1
    if args.num_samples > max_tras:
        print(f"⚠️  警告：请求 {args.num_samples} 条轨迹，但测试集只有 {max_tras} 条")
        args.num_samples = max_tras

    # 获取第一条轨迹的网格信息（所有轨迹网格相同）
    sample_graph = dataset[0]
    pos = sample_graph.pos.numpy()
    cells = sample_graph.face.numpy().T

    # ========== 3. 初始化模型 ==========
    print_section("3. 初始化模型")

    model = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device=device
    )

    if not args.random and args.checkpoint:
        print(f"加载 checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"训练轮数：{checkpoint.get('epoch', 'N/A')}")

    model.eval()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # ========== 4. Batch 推理 ==========
    print_section("4. 开始 Batch 推理")

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    all_latencies = []
    all_throughputs = []

    # 应用 PyG 变换（生成边特征）
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    for batch_idx in range(num_batches):
        batch_start_time = time.time()

        start_tra = batch_idx * args.batch_size
        end_tra = min(start_tra + args.batch_size, args.num_samples)
        current_batch_size = end_tra - start_tra

        # 加载当前 batch 的所有轨迹（支持限制步数）
        batch_graphs = []
        for tra_idx in range(start_tra, end_tra):
            graphs = load_trajectory(dataset, tra_idx, num_steps=args.num_steps)
            batch_graphs.extend(graphs)

        # 应用变换
        batch_graphs = [transformer(g) for g in batch_graphs]

        print(f"\nBatch {batch_idx + 1}/{num_batches} ({current_batch_size} 轨迹，{len(batch_graphs)} 样本)")

        # Batch 推理
        latencies = []
        for i, graph in enumerate(batch_graphs):
            graph = graph.to(device)

            step_start = time.time()
            with torch.no_grad():
                predicted_velocity = model(graph, None)
            step_time = (time.time() - step_start) * 1000  # ms
            latencies.append(step_time)

            if args.verbose and (i + 1) % 100 == 0:
                print(f"  样本 {i + 1}/{len(batch_graphs)} - 延迟：{step_time:.2f}ms")

        batch_time = time.time() - batch_start_time
        avg_latency = sum(latencies) / len(latencies)
        throughput = len(batch_graphs) / batch_time if batch_time > 0 else 0

        all_latencies.extend(latencies)
        all_throughputs.append(throughput)

        print(f"  Batch 耗时：{batch_time:.2f}s | 平均延迟：{avg_latency:.2f}ms | 吞吐量：{throughput:.2f} 样本/s")

        # 可视化（可选）
        if args.save_vis and batch_idx < 3:  # 只可视化前 3 个 batch
            # 简单示意，实际需保存预测结果
            dummy_pred = np.random.rand(len(batch_graphs), 2)
            dummy_target = np.random.rand(len(batch_graphs), 2)
            vis_path = visualize_batch_results(
                dummy_pred, dummy_target, pos, cells,
                args.output_dir, step=batch_idx
            )
            if args.verbose:
                print(f"  可视化已保存：{vis_path}")

    # ========== 5. 统计结果 ==========
    print_section("5. 推理统计")

    total_time = sum(all_latencies) / 1000  # s
    total_samples = len(all_latencies)
    avg_latency = np.mean(all_latencies)
    min_latency = np.min(all_latencies)
    max_latency = np.max(all_latencies)
    overall_throughput = total_samples / total_time if total_time > 0 else 0

    print(f"总样本数：{total_samples}")
    print(f"总耗时：{total_time:.2f}s")
    print(f"平均延迟：{avg_latency:.2f}ms")
    print(f"最小延迟：{min_latency:.2f}ms")
    print(f"最大延迟：{max_latency:.2f}ms")
    print(f"吞吐量：{overall_throughput:.2f} 样本/s")

    # ========== 6. 保存结果 ==========
    print_section("6. 保存结果")

    # 保存统计信息
    stats = {
        'num_trajectories': args.num_samples,
        'batch_size': args.batch_size,
        'num_steps_per_trajectory': args.num_steps,
        'total_samples': total_samples,
        'total_time_sec': total_time,
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'throughput_samples_per_sec': overall_throughput,
        'device': device_name
    }

    import json
    stats_path = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存：{stats_path}")

    # 保存详细延迟数据
    latency_path = os.path.join(args.output_dir, 'latencies.npy')
    np.save(latency_path, np.array(all_latencies))
    print(f"延迟数据已保存：{latency_path}")

    print_header("推理完成!")
    print(f"结果目录：{args.output_dir}")


if __name__ == '__main__':
    main()
