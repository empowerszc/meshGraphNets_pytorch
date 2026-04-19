#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MeshGraphNet 推理性能测试 - Worker 进程

支持两种模式:
1. CPU 多进程模式 (ARM 服务器)
2. GPU 单进程模式 (A100)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch

from model.simulator import Simulator
from dataset.fpc import FpcDataset


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='进程配置文件路径')
    parser.add_argument('--output', required=True, help='结果输出文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    process_id = config.get('process_id', 0)
    mode = config.get('mode', 'cpu')
    device_str = config.get('device', 'cpu')
    start_sample = config.get('start_sample', 0)
    end_sample = config.get('end_sample', 0)
    batch_size = config.get('batch_size', 4)
    split = config.get('split', 'test')
    checkpoint = config.get('checkpoint', '')

    # 线程配置（仅 CPU 模式）
    threads = config.get('threads', 1)
    core_list = config.get('core_list', '')
    mem_node = config.get('mem_node', 0)
    numa_id = config.get('numa_id', 0)

    # 设备
    if mode == 'gpu' or device_str.startswith('cuda'):
        device = torch.device(device_str if device_str else 'cuda:0')
        print(f"[进程 {process_id}] GPU 模式：设备={device}")
    else:
        device = torch.device('cpu')
        print(f"[进程 {process_id}] CPU 模式：NUMA={numa_id}, 线程={threads}")
        print(f"[进程 {process_id}] 绑定核心：[{core_list}]")
        print(f"[进程 {process_id}] 绑定内存节点：{mem_node}")
        print(f"[进程 {process_id}] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    print(f"[进程 {process_id}] 推理样本范围：{start_sample} - {end_sample}")
    print(f"[进程 {process_id}] Batch size: {batch_size}")

    # 检查是否有样本需要处理
    if start_sample > end_sample:
        print(f"[进程 {process_id}] 警告：样本范围为空，跳过推理")
        result = {
            'process_id': process_id,
            'samples_processed': 0,
            'elapsed_time': 0,
            'throughput': 0,
            'avg_latency_ms': 0,
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[进程 {process_id}] 结果已保存：{args.output}")
        return

    # 加载数据集
    print(f"[进程 {process_id}] 加载数据集...")
    dataset = FpcDataset(data_root='data', split=split)
    print(f"[进程 {process_id}] 数据集加载完成，总样本数：{len(dataset)}")

    # 初始化模型
    print(f"[进程 {process_id}] 初始化模型...")
    model = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device=str(device)
    )

    if checkpoint and os.path.exists(checkpoint):
        print(f"[进程 {process_id}] 加载 checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[进程 {process_id}] 训练轮数：{ckpt.get('epoch', 'N/A')}")
    else:
        if checkpoint:
            print(f"[进程 {process_id}] 警告：checkpoint 不存在，使用随机权重")

    model.eval()
    print(f"[进程 {process_id}] 模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # PyG 变换
    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])

    # ========== 开始推理 ==========
    print(f"[进程 {process_id}] 开始推理...")
    start_time = time.perf_counter()

    samples_processed = 0
    num_samples = end_sample - start_sample + 1

    # 计算 batch 数
    sample_indices = list(range(start_sample, end_sample + 1))
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = sample_indices[batch_start:batch_end]

        # 加载 batch 数据
        graphs = [dataset[i] for i in batch_indices]
        graphs = [transformer(g) for g in graphs]

        # Batch 推理
        batched_graph = Batch.from_data_list([g.to(device) for g in graphs])

        with torch.no_grad():
            _ = model(batched_graph, None)

        samples_processed += len(graphs)

        # 进度报告 (每 10 个 batch 或最后一个 batch)
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.perf_counter() - start_time
            throughput = samples_processed / elapsed if elapsed > 0 else 0
            print(f"[进程 {process_id}] 进度：{samples_processed}/{num_samples} "
                  f"({(batch_idx + 1) / num_batches * 100:.1f}%) | "
                  f"耗时：{elapsed:.2f}s | 吞吐量：{throughput:.2f} 样本/s")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    throughput = samples_processed / elapsed_time if elapsed_time > 0 else 0
    avg_latency = elapsed_time / samples_processed * 1000 if samples_processed > 0 else 0

    print(f"[进程 {process_id}] 推理完成!")
    print(f"[进程 {process_id}] 处理样本：{samples_processed}")
    print(f"[进程 {process_id}] 耗时：{elapsed_time:.2f}s")
    print(f"[进程 {process_id}] 吞吐量：{throughput:.2f} 样本/s")
    print(f"[进程 {process_id}] 平均延迟：{avg_latency:.2f}ms/样本")

    # 保存结果
    result = {
        'process_id': process_id,
        'mode': mode,
        'numa_id': numa_id,
        'core_list': core_list,
        'threads': threads,
        'samples_processed': samples_processed,
        'start_sample': start_sample,
        'end_sample': end_sample,
        'elapsed_time': elapsed_time,
        'throughput': throughput,
        'avg_latency_ms': avg_latency,
        'device': str(device)
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[进程 {process_id}] 结果已保存：{args.output}")


if __name__ == '__main__':
    main()
