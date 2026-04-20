#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Rollout V4 - 3D 张量预加载（最简洁版本）

核心思路:
1. 一次性将所有轨迹的所有时间步加载为 3D 张量
2. 按时间步切片：step_data[:, step, :]
3. 使用 PyG Batch 进行并行推理

数据结构:
    all_data: List[Tensor]  # 每个元素是一个轨迹的 [num_steps, N, 2] 速度张量
    all_pos: List[Tensor]   # 每个元素是一个轨迹的 [N, 2] 位置
    all_cells: List[Tensor] # 每个元素是一个轨迹的 [F, 3] 网格

    按时间步处理:
    for step in range(num_steps):
        step_velocities = [traj_data[step] for traj_data in all_data]
        # 构建图 → 打包 → 预测 → 拆分

优点:
    - 代码最简洁
    - 数据访问快速（内存连续）
    - 易于理解

缺点:
    - 内存占用大（不适合大数据集）
    - 不支持在线数据增强
"""

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data

from model.simulator import Simulator
from dataset.fpc import FpcDataset
from utils.utils import NodeType


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    print(f"\n--- {text} ---")


def get_boundary_mask(node_type: torch.Tensor) -> torch.Tensor:
    """创建边界条件掩码"""
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


def build_graph(pos: torch.Tensor, cells: torch.Tensor,
                velocity: torch.Tensor, node_type: torch.Tensor) -> Data:
    """
    从张量构建 PyG Data 对象

    Args:
        pos: [N, 2] 节点位置
        cells: [F, 3] 三角单元
        velocity: [N, 2] 速度
        node_type: [N, 1] 节点类型

    Returns:
        Data: PyG 图对象
    """
    graph = Data(
        x=torch.cat([node_type, velocity], dim=-1),  # [N, 1+2=3] → 实际是 [N, 11] 需要 one-hot
        pos=pos,
        face=cells.t().contiguous()  # [3, F]
    )
    graph.y = velocity  # 目标速度（占位，实际会被替换）
    return graph


def main():
    args = parse_args()

    # ========== 1. 配置 ==========
    print_header("Batch Rollout V4 - 3D 张量预加载")

    print_section("1. 配置")
    print(f"轨迹数：{args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备：{device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 2. 数据集 - 预加载所有数据 ==========
    print_section("2. 预加载数据集")

    dataset = FpcDataset(data_root='data', split=args.split)
    num_steps = dataset.num_sampes_per_tra

    # 预加载所有轨迹的数据为 3D 张量
    print("正在加载数据到内存...")
    start_load = time.time()

    all_velocities = []      # List[[num_steps, N_i, 2]] - 每条轨迹的速度
    all_positions = []       # List[[N_i, 2]] - 每条轨迹的位置
    all_cells = []           # List[[F_i, 3]] - 每条轨迹的网格
    all_node_types = []      # List[[N_i, 1]] - 每条轨迹的节点类型
    boundary_masks = []      # List[[N_i]] - 每条轨迹的边界掩码

    for tra_idx in range(args.num_samples):
        # 获取该轨迹的速度数据
        start_idx = tra_idx * num_steps
        velocities = []
        for step in range(num_steps):
            graph = dataset[start_idx + step]
            if step == 0:
                all_positions.append(graph.pos.clone())
                all_cells.append(graph.face.t().clone())
                all_node_types.append(graph.x[:, 0:1].clone())
                # 计算边界掩码
                node_type = graph.x[:, 0].long()
                predict_mask = torch.logical_or(
                    node_type == NodeType.NORMAL,
                    node_type == NodeType.OUTFLOW
                )
                boundary_masks.append(torch.logical_not(predict_mask))
            velocities.append(graph.y.clone())

        # 堆叠为 [num_steps, N, 2]
        vel_tensor = torch.stack(velocities, dim=0)
        all_velocities.append(vel_tensor)

    load_time = time.time() - start_load
    print(f"加载完成：{load_time:.2f}s, {len(all_velocities)} 条轨迹")

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

    # PyG 变换
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # ========== 4. Rollout ==========
    print_section("4. Rollout 推理")

    start_time = time.time()

    # 当前速度（会被更新）
    current_velocities = [vel[0].clone() for vel in all_velocities]  # [N, 2]

    all_predicteds = [[] for _ in range(args.num_samples)]
    all_targets = [[] for _ in range(args.num_samples)]

    # 按时间步循环（串行）
    for step in range(num_steps):
        # 获取所有轨迹的当前步速度
        step_velocities = [all_velocities[tra_idx][step] for tra_idx in range(args.num_samples)]

        # 分 batch 处理
        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_samples)

            # 构建当前 batch 的图（每条轨迹用自己的网格）
            batch_graphs = []
            for tra_idx in range(batch_start, batch_end):
                graph = Data(
                    x=torch.cat([
                        all_node_types[tra_idx],  # [N_i, 1]
                        current_velocities[tra_idx],  # [N_i, 2]
                    ], dim=-1),
                    pos=all_positions[tra_idx],
                    face=all_cells[tra_idx].t().contiguous()
                )
                graph.y = step_velocities[tra_idx]  # 真值
                batch_graphs.append(graph)

            # 应用变换
            batch_graphs = [transformer(g).to(device) for g in batch_graphs]

            # 自回归输入：更新速度
            for i, graph in enumerate(batch_graphs):
                tra_idx = batch_start + i
                if step > 0:
                    graph.x[:, 1:3] = current_velocities[tra_idx].to(device)

            # 打包成 batch
            batched = Batch.from_data_list(batch_graphs)

            # 模型预测
            with torch.no_grad():
                batched_pred = model(batched, None)

            # 拆分结果（按每条轨迹的节点数）
            num_nodes_list = [g.num_nodes for g in batch_graphs]
            preds_split = torch.split(batched_pred, num_nodes_list, dim=0)

            # 边界约束并更新
            for i, pred in enumerate(preds_split):
                tra_idx = batch_start + i

                pred = pred.clone()
                pred[boundary_masks[tra_idx].to(device)] = step_velocities[tra_idx].to(device)[boundary_masks[tra_idx].to(device)]

                # 更新下一帧输入
                current_velocities[tra_idx] = pred.cpu()

                # 保存结果
                all_predicteds[tra_idx].append(pred.cpu().numpy())
                all_targets[tra_idx].append(step_velocities[tra_idx].numpy())

        if args.verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    elapsed = time.time() - start_time

    # ========== 5. 统计 ==========
    print_section("5. 统计")

    print(f"总耗时：{elapsed:.2f}s (含加载 {load_time:.2f}s)")

    rmse_curves = []
    for i in range(args.num_samples):
        rmse = compute_rollout_error(all_predicteds[i], all_targets[i])
        rmse_curves.append(rmse)

    rmse_array = np.array(rmse_curves)
    print(f"RMSE: 初始={rmse_array[:, 0].mean():.2e}, 最终={rmse_array[:, -1].mean():.2e}")

    # ========== 6. 保存 ==========
    print_section("6. 保存结果")

    if args.save_results:
        os.makedirs(f'{args.output_dir}/results', exist_ok=True)
        crds = pos.cpu().numpy()

        for i in range(args.num_samples):
            result = [np.stack(all_predicteds[i]), np.stack(all_targets[i])]
            with open(f'{args.output_dir}/results/result{i}.pkl', 'wb') as f:
                pickle.dump([result, crds], f)

    print_header("完成!")


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Rollout V4 - 3D Tensor')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--random', action='store_true', default=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str,
                        default=f'outputs/batch_rollout_v4/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
