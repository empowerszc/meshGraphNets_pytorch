#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Rollout V3 - 自定义 TrajectoryDataset

核心思路:
1. 封装原始 dataset 为 TrajectoryDataset，按轨迹索引
2. 实现 __getitem__ 返回整条轨迹的所有时间步
3. 使用 collate_fn 按时间步重组 batch

数据结构:
    TrajectoryDataset[num_trajectories]
    __getitem__(tra_idx) → List[Graph] (该轨迹的所有时间步)

    collate_fn 将 batch 条轨迹重组为 num_steps 个 batch:
    输入：[trajectory_0, trajectory_1, ..., trajectory_{batch-1}]
    其中每个 trajectory_i = [step_0, step_1, ..., step_{N-1}]

    输出：生成器，每次 yield 一个时间步的 batch
    yield 1: [step_0 of tra_0, step_0 of tra_1, ...]
    yield 2: [step_1 of tra_0, step_1 of tra_1, ...]
    ...

优点:
    - 封装良好，接口清晰
    - 可按轨迹并行加载数据
    - 易于扩展（如添加数据增强）

缺点:
    - collate_fn 不能是生成器（PyTorch 限制）
    - 需要手动管理时间步循环
"""

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader
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


class TrajectoryDataset(Dataset):
    """
    按轨迹组织的 Dataset 包装器

    将平铺的 dataset 重新组织为按轨迹索引:
    - 原始：dataset[tra_idx * num_steps + step]
    - 包装后：traj_dataset[tra_idx] → 该轨迹的所有时间步
    """

    def __init__(self, dataset, num_trajectories, transformer=None):
        """
        Args:
            dataset: 原始 FpcDataset
            num_trajectories: 要加载的轨迹数量
            transformer: PyG 变换（可选）
        """
        self.dataset = dataset
        self.num_trajectories = num_trajectories
        self.num_steps = dataset.num_sampes_per_tra
        self.transformer = transformer

    def __len__(self):
        return self.num_trajectories

    def get_step(self, tra_idx: int, step: int) -> Data:
        """获取指定轨迹的指定时间步"""
        idx = tra_idx * self.num_steps + step
        graph = self.dataset[idx]
        if self.transformer:
            graph = self.transformer(graph)
        return graph

    def get_trajectory(self, tra_idx: int) -> List[Data]:
        """获取整条轨迹的所有时间步"""
        return [self.get_step(tra_idx, step) for step in range(self.num_steps)]

    def __getitem__(self, tra_idx) -> List[Data]:
        """返回整条轨迹（所有时间步的图列表）"""
        return self.get_trajectory(tra_idx)


class StepCollator:
    """
    按时间步重组 batch 的 collator

    用途：将多条轨迹的图按时间步分组
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, batch_trajectories):
        """
        Args:
            batch_trajectories: List[List[Data]]
                外层：batch 条轨迹
                内层：每条轨迹的 num_steps 个图

        Returns:
            List[List[Data]]: 按时间步重组
                [
                    [tra0_step0, tra1_step0, ...],  # step 0 的 batch
                    [tra0_step1, tra1_step1, ...],  # step 1 的 batch
                    ...
                ]
        """
        # 按时间步重组
        step_batches = []
        for step in range(self.num_steps):
            step_graphs = [traj[step] for traj in batch_trajectories]
            step_batches.append(step_graphs)

        return step_batches


def main():
    args = parse_args()

    # ========== 1. 配置 ==========
    print_header("Batch Rollout V3 - TrajectoryDataset")

    print_section("1. 配置")
    print(f"轨迹数：{args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备：{device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 2. 数据集 ==========
    print_section("2. 加载数据集")

    dataset = FpcDataset(data_root='data', split=args.split)
    num_steps = dataset.num_sampes_per_tra

    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # 创建 TrajectoryDataset
    traj_dataset = TrajectoryDataset(
        dataset=dataset,
        num_trajectories=args.num_samples,
        transformer=transformer
    )

    print(f"轨迹数据集：{len(traj_dataset)} 条轨迹 × {num_steps} 步")

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

    # ========== 4. 预计算边界掩码 ==========
    boundary_masks = []
    for tra_idx in range(args.num_samples):
        graph = dataset[tra_idx * num_steps]
        mask = get_boundary_mask(graph)
        boundary_masks.append(mask)

    # ========== 5. Rollout ==========
    print_section("4. Rollout 推理")

    start_time = time.time()
    all_predicteds = [[] for _ in range(args.num_samples)]
    all_targets = [[] for _ in range(args.num_samples)]
    predicted_prev = [None] * args.num_samples

    # 创建 DataLoader（按轨迹加载）
    loader = DataLoader(
        traj_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x  # 保持原始结构
    )

    # 获取所有轨迹的数据（一次性）
    all_trajectories = []
    for batch_trajectories in loader:
        all_trajectories.extend(batch_trajectories)

    # 按时间步循环（串行）
    for step in range(num_steps):
        # 收集所有轨迹的当前时间步
        step_graphs = [traj[step] for traj in all_trajectories]

        # 分 batch 处理
        for batch_start in range(0, len(step_graphs), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(step_graphs))
            batch_graphs = step_graphs[batch_start:batch_end]

            # 移到 device
            batch_graphs = [g.to(device) for g in batch_graphs]

            # 自回归输入
            for i, graph in enumerate(batch_graphs):
                tra_idx = batch_start + i
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
                tra_idx = batch_start + i

                pred = pred.clone()
                pred[boundary_masks[tra_idx].to(device)] = graph.y.to(device)[boundary_masks[tra_idx].to(device)]

                predicted_prev[tra_idx] = pred
                all_predicteds[tra_idx].append(pred.cpu().numpy())
                all_targets[tra_idx].append(graph.y.cpu().numpy())

        if args.verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    elapsed = time.time() - start_time

    # ========== 6. 统计 ==========
    print_section("5. 统计")

    print(f"总耗时：{elapsed:.2f}s")

    rmse_curves = []
    for i in range(args.num_samples):
        rmse = compute_rollout_error(all_predicteds[i], all_targets[i])
        rmse_curves.append(rmse)

    rmse_array = np.array(rmse_curves)
    print(f"RMSE: 初始={rmse_array[:, 0].mean():.2e}, 最终={rmse_array[:, -1].mean():.2e}")

    # ========== 7. 保存 ==========
    print_section("6. 保存结果")

    if args.save_results:
        os.makedirs(f'{args.output_dir}/results', exist_ok=True)
        crds = dataset[0].pos.cpu().numpy()

        for i in range(args.num_samples):
            result = [np.stack(all_predicteds[i]), np.stack(all_targets[i])]
            with open(f'{args.output_dir}/results/result{i}.pkl', 'wb') as f:
                pickle.dump([result, crds], f)

    print_header("完成!")


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Rollout V3 - TrajectoryDataset')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--random', action='store_true', default=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str,
                        default=f'outputs/batch_rollout_v3/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
