#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本 - 比较不同 batch rollout 实现的输出一致性

使用方法:
    1. 先导出测试模型：python export_test_model.py
    2. 运行各版本 rollout:
       python batch_rollout.py --checkpoint checkpoints/test_random.pth --num-samples 2 --batch-size 1 --save-results
       python batch_rollout.py --checkpoint checkpoints/test_random.pth --num-samples 2 --batch-size 2 --save-results
       python batch_rollout_v2.py --checkpoint checkpoints/test_random.pth --num-samples 2 --batch-size 1 --save-results
       python batch_rollout_v2.py --checkpoint checkpoints/test_random.pth --num-samples 2 --batch-size 2 --save-results
       ...
    3. 运行本脚本比较结果：python verify_rollouts.py
"""

import os
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np


def find_latest_rollout_results(base_dir='outputs'):
    """查找最新的 rollout 结果目录"""
    output_dirs = {
        'v1': [],
        'v2': [],
        'v3': [],
        'v4': [],
    }

    base_path = Path(base_dir)
    # 遍历版本目录 (batch_rollout, batch_rollout_v2, etc.)
    for version_dir in base_path.iterdir():
        if not version_dir.is_dir() or str(version_dir).startswith('._'):
            continue

        # 遍历每个版本目录下的时间戳子目录
        for d in version_dir.iterdir():
            if not d.is_dir() or str(d).startswith('._'):
                continue

            if 'batch_rollout_v2' in str(version_dir):
                output_dirs['v2'].append(d)
            elif 'batch_rollout_v3' in str(version_dir):
                output_dirs['v3'].append(d)
            elif 'batch_rollout_v4' in str(version_dir):
                output_dirs['v4'].append(d)
            elif 'batch_rollout' in str(version_dir) and 'v' not in str(version_dir):
                output_dirs['v1'].append(d)

    # 按时间排序，取最新的
    results = {}
    for key, dirs in output_dirs.items():
        if dirs:
            latest = max(dirs, key=lambda d: d.stat().st_mtime)
            results_file = latest / 'results' / 'result0.pkl'
            if results_file.exists():
                results[key] = latest

    return results


def load_results(results_dir):
    """加载结果文件"""
    results = []
    results_path = Path(results_dir) / 'results'

    i = 0
    while True:
        pkl_file = results_path / f'result{i}.pkl'
        if not pkl_file.exists():
            break
        with open(pkl_file, 'rb') as f:
            data, crds = pickle.load(f)
            results.append({
                'predicteds': data[0],  # [num_steps, N, 2]
                'targets': data[1],
                'crds': crds
            })
        i += 1

    return results


def compare_results(results_v1, results_vx, name_vx, tol=1e-5):
    """比较两个结果是否一致"""
    print(f"\n{'='*60}")
    print(f"对比：V1 (基准) vs {name_vx}")
    print(f"{'='*60}")

    all_pass = True

    for i, (r1, rx) in enumerate(zip(results_v1, results_vx)):
        pred1 = r1['predicteds']
        predx = rx['predicteds']

        # 检查形状
        if len(pred1) != len(predx):
            print(f"  轨迹 {i}: ❌ 步数不一致 ({len(pred1)} vs {len(predx)})")
            all_pass = False
            continue

        # 检查数值
        max_diff = np.max(np.abs(pred1 - predx))
        mean_diff = np.mean(np.abs(pred1 - predx))

        if max_diff < tol:
            print(f"  轨迹 {i}: ✅ 完全一致 (max_diff={max_diff:.2e})")
        else:
            print(f"  轨迹 {i}: ❌ 差异超出容差 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
            all_pass = False

    return all_pass


def compute_statistics(results):
    """计算统计信息"""
    stats = []
    for r in results:
        predicteds = np.stack(r['predicteds'])
        targets = np.stack(r['targets'])

        # 计算每步 MSE
        mse = np.mean((predicteds - targets) ** 2)
        rmse = np.sqrt(mse)

        # 计算边界节点误差
        # （假设边界节点预测值应等于真值）

        stats.append({
            'rmse': rmse,
            'steps': len(predicteds),
            'nodes': predicteds.shape[1]
        })

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='outputs')
    parser.add_argument('--tol', type=float, default=1e-5, help='数值容差')
    args = parser.parse_args()

    print("="*60)
    print("  Batch Rollout 结果验证")
    print("="*60)

    # 查找结果
    results_dirs = find_latest_rollout_results(args.base_dir)

    if not results_dirs:
        print("❌ 未找到任何 rollout 结果")
        print("请先运行各版本的 rollout 脚本并保存结果")
        return

    print(f"\n找到以下版本的结果:")
    for key, path in results_dirs.items():
        print(f"  V{key}: {path}")

    # 加载 V1 作为基准
    if 'v1' not in results_dirs:
        print("❌ 未找到 V1 (基准) 结果")
        return

    print(f"\n加载 V1 结果...")
    results_v1 = load_results(results_dirs['v1'])
    print(f"  加载了 {len(results_v1)} 条轨迹")

    # 统计信息
    stats_v1 = compute_statistics(results_v1)
    print(f"\nV1 统计信息:")
    for i, s in enumerate(stats_v1):
        print(f"  轨迹 {i}: RMSE={s['rmse']:.2e}, Steps={s['steps']}, Nodes={s['nodes']}")

    # 比较其他版本
    all_pass = True
    for key in ['v2', 'v3', 'v4']:
        if key in results_dirs:
            print(f"\n加载 V{key} 结果...")
            results_vx = load_results(results_dirs[key])
            print(f"  加载了 {len(results_vx)} 条轨迹")

            if len(results_vx) != len(results_v1):
                print(f"❌ V{key} 轨迹数与 V1 不一致")
                all_pass = False
                continue

            passed = compare_results(results_v1, results_vx, f"V{key}", args.tol)
            if not passed:
                all_pass = False

    # 总结
    print(f"\n{'='*60}")
    if all_pass:
        print("✅ 所有版本结果一致！验证通过。")
    else:
        print("❌ 部分版本结果不一致，请检查代码。")
    print(f"{'='*60}")

    # 生成报告
    report_path = Path(args.base_dir) / f"verify_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Batch Rollout 验证报告\n")
        f.write(f"时间：{datetime.now()}\n")
        f.write(f"容差：{args.tol}\n")
        f.write(f"结果：{'通过' if all_pass else '失败'}\n")
    print(f"\n报告已保存：{report_path}")


if __name__ == '__main__':
    main()
