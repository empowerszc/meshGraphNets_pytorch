#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Rollout V2 - 使用 Subset + DataLoader 按时间步分组

核心思路:
1. 使用 Subset 为每个时间步创建"视图"
2. 使用 DataLoader 按 batch_size 加载同一时间步的多条轨迹
3. 时间步之间串行，轨迹之间并行

数据结构:
    原始数据：dataset[0], dataset[1], ..., dataset[N*steps-1]
    按时间步重组:
        step 0: Subset(dataset, [0, steps, 2*steps, ...])
        step 1: Subset(dataset, [1, steps+1, 2*steps+1, ...])
        ...

优点:
    - 使用标准 PyTorch API (Subset, DataLoader)
    - 支持 DataLoader 的多进程加载 (num_workers > 0)
    - 代码清晰，易于理解

缺点:
    - 需要为每个时间步创建 Subset
    - 索引计算略复杂
"""

import argparse
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import Subset, DataLoader
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


def get_boundary_mask(graph):
    """创建边界条件掩码"""
    node_type = graph.x[:, 0].long()
    predict_mask = torch.logical_or(
        node_type == NodeType.NORMAL,
        node_type == NodeType.OUTFLOW
    )
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
    print_header("Batch Rollout V2 - Subset + DataLoader")

    print_section("1. 配置")
    print(f"轨迹数：{args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"DataLoader workers: {args.workers}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备：{device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 2. 数据集 ==========
    print_section("2. 加载数据集")

    dataset = FpcDataset(data_root='data', split=args.split)
    num_steps = dataset.num_sampes_per_tra
    print(f"每条轨迹时间步数：{num_steps}")

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

    model.eval()

    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # ========== 4. 按时间步创建 Subset ==========
    # 关键：为每个时间步创建一个 Subset，包含所有轨迹的该时间步数据
    # 例如：step_subsets[0] 包含所有轨迹的第 0 步
    step_subsets = []
    for step in range(num_steps):
        # 计算该时间步在所有轨迹中的全局索引
        # 轨迹 0 的 step, 轨迹 1 的 step, 轨迹 2 的 step, ...
        indices = [tra_idx * num_steps + step for tra_idx in range(args.num_samples)]
        step_subsets.append(Subset(dataset, indices))

    print(f"创建了 {len(step_subsets)} 个时间步 Subset")

    # ========== 5. 预计算边界掩码 ==========
    boundary_masks = []
    for tra_idx in range(args.num_samples):
        graph = dataset[tra_idx * num_steps]
        mask = get_boundary_mask(graph)
        boundary_masks.append(mask)

    # ========== 6. Rollout ==========
    print_section("4. Rollout 推理")

    start_time = time.time()
    all_predicteds = [[] for _ in range(args.num_samples)]
    all_targets = [[] for _ in range(args.num_samples)]
    predicted_prev = [None] * args.num_samples

    # 按时间步串行循环
    for step in range(num_steps):
        # 使用 DataLoader 加载当前时间步的所有轨迹
        step_loader = DataLoader(
            step_subsets[step],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=lambda x: x  # 不使用默认 collate（PyG Data 不支持）
        )

        batch_start_idx = 0  # 当前 step 的起始轨迹索引

        # 按 batch 处理
        for batch_graphs in step_loader:
            # 应用变换并移到 device
            batch_graphs = [transformer(g).to(device) for g in batch_graphs]

            # 自回归输入：用上一帧预测替换当前速度
            for i, graph in enumerate(batch_graphs):
                tra_idx = batch_start_idx + i
                if predicted_prev[tra_idx] is not None:
                    graph.x[:, 1:3] = predicted_prev[tra_idx].to(device)

            # 打包成 batch
            batched = Batch.from_data_list(batch_graphs)

            # 模型预测
            with torch.no_grad():
                batched_pred = model(batched, None)

            # 拆分结果
            num_nodes_list = [g.num_nodes for g in batch_graphs]
            preds_split = torch.split(batched_pred, num_nodes_list, dim=0)

            # 边界约束并保存
            for i, (pred, graph) in enumerate(zip(preds_split, batch_graphs)):
                tra_idx = batch_start_idx + i

                pred = pred.clone()
                pred[boundary_masks[tra_idx].to(device)] = graph.y.to(device)[boundary_masks[tra_idx].to(device)]

                predicted_prev[tra_idx] = pred
                all_predicteds[tra_idx].append(pred.cpu().numpy())
                all_targets[tra_idx].append(graph.y.cpu().numpy())

            batch_start_idx += len(batch_graphs)

        if args.verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    elapsed = time.time() - start_time

    # ========== 7. 统计 ==========
    print_section("5. 统计")

    print(f"总耗时：{elapsed:.2f}s")
    print(f"平均每步：{elapsed / num_steps * 1000:.2f}ms")

    rmse_curves = []
    for i in range(args.num_samples):
        rmse = compute_rollout_error(all_predicteds[i], all_targets[i])
        rmse_curves.append(rmse)

    rmse_array = np.array(rmse_curves)
    print(f"\nRMSE: 初始={rmse_array[:, 0].mean():.2e}, 最终={rmse_array[:, -1].mean():.2e}")

    # ========== 8. 保存 ==========
    print_section("6. 保存结果")

    if args.save_results:
        os.makedirs(f'{args.output_dir}/results', exist_ok=True)
        crds = dataset[0].pos.cpu().numpy()

        for i in range(args.num_samples):
            result = [np.stack(all_predicteds[i]), np.stack(all_targets[i])]
            with open(f'{args.output_dir}/results/result{i}.pkl', 'wb') as f:
                pickle.dump([result, crds], f)
            print(f"  轨迹 {i}: result{i}.pkl")

    print_header("完成!")


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Rollout V2 - Subset + DataLoader')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0,
                        help='DataLoader 加载进程数 (0=主进程)')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--random', action='store_true', default=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str,
                        default=f'outputs/batch_rollout_v2/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
