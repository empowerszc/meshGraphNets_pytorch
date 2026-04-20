#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Rollout - MeshGraphNet 批量 Rollout 推理脚本

设计思路:
- rollout 的串行性只在时间维度（第 t 步依赖 t-1 步）
- 不同轨迹之间完全独立，可以在同一时间步并行处理
- 按时间步组织数据：每个时间步将所有轨迹打包成一个 batch

数据结构:
    trajectories[tra_idx][step] = graph

    按 step 重组:
    step 0: [tra0_step0, tra1_step0, ...] → Batch → model.forward() → split
    step 1: [tra0_step1, tra1_step1, ...] → Batch → model.forward() → split
    ...

使用方法:
    python batch_rollout.py --num-samples 10 --batch-size 2
"""

import argparse
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch

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
    parser = argparse.ArgumentParser(description='MeshGraphNet Batch Rollout')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='推理的轨迹数量')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批处理大小（每个时间步分几个 batch 处理）')
    parser.add_argument('--checkpoint', type=str,
                        help='模型 checkpoint 路径')
    parser.add_argument('--random', action='store_true', default=True,
                        help='使用随机权重')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'valid', 'train'])
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output-dir', type=str,
                        default=f'outputs/batch_rollout/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--save-results', action='store_true',
                        help='保存完整推理结果 (pkl 格式，与 rollout.py 兼容)')
    parser.add_argument('--verbose', action='store_true',
                        help='详细日志')
    return parser.parse_args()


def get_boundary_mask(graph):
    """创建边界条件掩码：True 表示需要固定的节点（非预测节点）"""
    node_type = graph.x[:, 0].long()
    # 需要预测的节点：NORMAL + OUTFLOW
    predict_mask = torch.logical_or(
        node_type == NodeType.NORMAL,
        node_type == NodeType.OUTFLOW
    )
    # 需要固定的节点：边界、入口、障碍物等
    return torch.logical_not(predict_mask)


def compute_rollout_error(predicteds, targets):
    """计算 rollout 累积 RMSE"""
    predicteds = np.stack(predicteds)
    targets = np.stack(targets)
    squared_diff = np.square(predicteds - targets)
    rmse = np.sqrt(np.cumsum(np.mean(squared_diff.reshape(len(predicteds), -1), axis=1)) /
                   np.arange(1, len(predicteds) + 1))
    return rmse


def main():
    args = parse_args()

    # ========== 1. 配置 ==========
    print_header("MeshGraphNet Batch Rollout")

    print_section("1. 配置")
    print(f"轨迹数：{args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"数据集：{args.split}")

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"设备：{device}")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 2. 数据集 ==========
    print_section("2. 加载数据集")

    dataset = FpcDataset(data_root='data', split=args.split)
    num_steps = dataset.num_sampes_per_tra

    print(f"总样本数：{len(dataset)}")
    print(f"每条轨迹时间步数：{num_steps}")

    # 检查轨迹数合法性
    max_tras = len(dataset.meta['indices']) - 1
    if args.num_samples > max_tras:
        print(f"⚠️  警告：请求 {args.num_samples} 条轨迹，但数据集只有 {max_tras} 条")
        args.num_samples = max_tras

    # ========== 3. 模型 ==========
    print_section("3. 初始化模型")

    model = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device=str(device)
    )

    if not args.random and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"加载 checkpoint: {args.checkpoint}")

    model.eval()
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

    # PyG 变换
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # ========== 4. 准备数据 ==========
    # 按轨迹组织数据的索引
    trajectories_indices = []
    for tra_idx in range(args.num_samples):
        start_idx = tra_idx * num_steps
        end_idx = start_idx + num_steps
        trajectories_indices.append(list(range(start_idx, end_idx)))

    # 缓存每条轨迹的边界掩码（所有时间步共用）
    boundary_masks = []
    for tra_idx in range(args.num_samples):
        graph = dataset[trajectories_indices[tra_idx][0]]
        mask = get_boundary_mask(graph)
        boundary_masks.append(mask)

    # ========== 5. Batch Rollout ==========
    print_section("4. Rollout 推理")

    start_time = time.time()

    # 存储结果：每条轨迹的预测和真值序列
    all_predicteds = [[] for _ in range(args.num_samples)]
    all_targets = [[] for _ in range(args.num_samples)]

    # 存储每条轨迹的上一帧预测速度（用于自回归输入）
    predicted_prev = [None] * args.num_samples

    # 按时间步循环（串行）
    for step in range(num_steps):
        # 收集所有轨迹的当前时间步数据
        step_graphs = []
        for tra_idx in range(args.num_samples):
            graph = dataset[trajectories_indices[tra_idx][step]]
            graph = transformer(graph)
            step_graphs.append(graph)

        # 自回归输入：用上一帧的预测替换当前输入速度
        for tra_idx, graph in enumerate(step_graphs):
            if predicted_prev[tra_idx] is not None:
                graph.x[:, 1:3] = predicted_prev[tra_idx].to(device)

        # 分 batch 处理（当轨迹数较多时）
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, args.num_samples)
            batch_graphs = step_graphs[start_idx:end_idx]

            # 打包成 batch
            batched = Batch.from_data_list([g.to(device) for g in batch_graphs])

            # 模型预测
            with torch.no_grad():
                batched_pred = model(batched, None)  # [N_batch, 2]

            # 拆分结果
            num_nodes_list = [g.num_nodes for g in batch_graphs]
            preds_split = torch.split(batched_pred, num_nodes_list, dim=0)

            # 应用边界约束并保存
            for i, (pred, graph) in enumerate(zip(preds_split, batch_graphs)):
                tra_idx = start_idx + i

                # 边界条件约束
                pred = pred.clone()
                pred[boundary_masks[tra_idx].to(device)] = graph.y.to(device)[boundary_masks[tra_idx].to(device)]

                # 更新下一帧的输入
                predicted_prev[tra_idx] = pred

                # 保存结果
                all_predicteds[tra_idx].append(pred.cpu().numpy())
                all_targets[tra_idx].append(graph.y.cpu().numpy())

        if args.verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    elapsed = time.time() - start_time

    # ========== 6. 统计 ==========
    print_section("5. 统计")

    print(f"总耗时：{elapsed:.2f}s")
    print(f"平均每轨迹：{elapsed / args.num_samples:.2f}s")
    print(f"平均每步：{elapsed / (args.num_samples * num_steps) * 1000:.2f}ms")

    # RMSE 统计
    rmse_curves = []
    for i in range(args.num_samples):
        rmse = compute_rollout_error(all_predicteds[i], all_targets[i])
        rmse_curves.append(rmse)
        print(f"  轨迹 {i}: 初始={rmse[0]:.2e}, 最终={rmse[-1]:.2e}")

    rmse_array = np.array(rmse_curves)
    print(f"\nRMSE 统计 (平均 ± 标准差):")
    print(f"  初始步：{rmse_array[:, 0].mean():.2e} ± {rmse_array[:, 0].std():.2e}")
    print(f"  最终步：{rmse_array[:, -1].mean():.2e} ± {rmse_array[:, -1].std():.2e}")

    # ========== 7. 保存 ==========
    print_section("6. 保存结果")

    # 获取网格坐标（所有轨迹共用）
    crds = dataset[0].pos.cpu().numpy()

    # 统计信息
    stats = {
        'num_trajectories': args.num_samples,
        'batch_size': args.batch_size,
        'num_steps': num_steps,
        'elapsed_time_sec': elapsed,
        'avg_rmse_final': float(rmse_array[:, -1].mean()),
        'device': str(device)
    }

    import json
    with open(f'{args.output_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息：{args.output_dir}/statistics.json")

    # 完整结果（与 rollout.py 兼容的格式）
    if args.save_results:
        os.makedirs(f'{args.output_dir}/results', exist_ok=True)

        for i in range(args.num_samples):
            # 堆叠为 [num_steps, N, 2]
            result = [np.stack(all_predicteds[i]), np.stack(all_targets[i])]

            # 保存为 result/result{index}.pkl，与 rollout.py 格式一致
            pkl_path = f'{args.output_dir}/results/result{i}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump([result, crds], f)
            print(f"  轨迹 {i}: {pkl_path}")

        # 额外保存汇总文件（包含所有轨迹的 RMSE）
        summary = {
            'results': [
                {'index': i, 'rmse': rmse_curves[i].tolist()}
                for i in range(args.num_samples)
            ],
            'crds': crds
        }
        with open(f'{args.output_dir}/results/summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        print(f"  汇总：{args.output_dir}/results/summary.pkl")

    print_header("完成!")


if __name__ == '__main__':
    main()
