#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MeshGraphNet Rollout 结果比较工具

专门用于比较 rollout 推理生成的 PKL 结果文件。

PKL 文件格式:
    [result, crds] = pickle.load(f)
    result = [predicteds, targets]  # List[[num_steps, N, 2], [num_steps, N, 2]]
    crds = coordinates  # [N, 2]

使用方法:
    # 比较两个文件
    python compare_rollout_results.py outputs/v1/results/result0.pkl outputs/v2/results/result0.pkl

    # 比较两个目录中的所有结果文件
    python compare_rollout_results.py outputs/v1/results/ outputs/v2/results/

    # 指定容差
    python compare_rollout_results.py file1.pkl file2.pkl --tol 1e-4

    # 详细输出（打印误差统计）
    python compare_rollout_results.py file1.pkl file2.pkl --verbose
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_rollout_pkl(file_path: str) -> Dict:
    """
    加载 rollout 结果 PKL 文件

    Returns:
        dict: {
            'predicteds': np.ndarray [num_steps, N, 2],
            'targets': np.ndarray [num_steps, N, 2],
            'coordinates': np.ndarray [N, 2]
        }
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    result, crds = data
    predicteds, targets = result

    return {
        'predicteds': np.stack(predicteds) if isinstance(predicteds, list) else predicteds,
        'targets': np.stack(targets) if isinstance(targets, list) else targets,
        'coordinates': crds
    }


def compare_rollout_results(
    data1: Dict,
    data2: Dict,
    tol: float = 1e-5,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    比较两个 rollout 结果

    Args:
        data1, data2: load_rollout_pkl 返回的字典
        tol: 数值容差
        verbose: 是否输出详细统计

    Returns:
        (是否一致，差异描述)
    """
    messages = []
    all_match = True

    # 比较 predicteds
    pred1 = data1['predicteds']
    pred2 = data2['predicteds']

    if pred1.shape != pred2.shape:
        messages.append(f"❌ predicteds 形状不同：{pred1.shape} vs {pred2.shape}")
        all_match = False
    else:
        max_diff = np.max(np.abs(pred1 - pred2))
        mean_diff = np.mean(np.abs(pred1 - pred2))

        if max_diff < tol:
            messages.append(f"✅ predicteds: 完全一致 (max_diff={max_diff:.2e})")
        else:
            messages.append(f"❌ predicteds: 差异超出容差 (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
            all_match = False

        if verbose:
            # 每步误差统计
            print("\n每步最大误差:")
            step_max_diffs = np.max(np.abs(pred1 - pred2), axis=(1, 2))
            for step in [0, 1, 10, 50, 100, 200, 400, len(step_max_diffs)-1]:
                if step < len(step_max_diffs):
                    print(f"  Step {step}: {step_max_diffs[step]:.2e}")

    # 比较 targets
    tgt1 = data1['targets']
    tgt2 = data2['targets']

    if tgt1.shape != tgt2.shape:
        messages.append(f"❌ targets 形状不同：{tgt1.shape} vs {tgt2.shape}")
        all_match = False
    else:
        max_diff = np.max(np.abs(tgt1 - tgt2))
        if max_diff < tol:
            messages.append(f"✅ targets: 完全一致 (max_diff={max_diff:.2e})")
        else:
            messages.append(f"❌ targets: 差异超出容差 (max_diff={max_diff:.2e})")
            all_match = False

    # 比较 coordinates (仅供参考，不影响一致性判断)
    # 注意：不同轨迹的网格可能不同，coordinates 形状不同是正常的
    crd1 = data1['coordinates']
    crd2 = data2['coordinates']

    if crd1.shape != crd2.shape:
        # 形状不同不代表错误，只是说明是不同的轨迹
        messages.append(f"⚠️  coordinates 形状不同：{crd1.shape} vs {crd2.shape} (可能是不同轨迹)")
        # 不设置 all_match = False
    else:
        max_diff = np.max(np.abs(crd1 - crd2))
        if max_diff < tol:
            messages.append(f"✅ coordinates: 完全一致 (max_diff={max_diff:.2e})")
        else:
            messages.append(f"⚠️  coordinates: 存在差异 (max_diff={max_diff:.2e})")
            # 坐标差异不影响一致性判断

    # RMSE 统计
    if verbose:
        def compute_rmse(predicteds, targets):
            squared_diff = np.square(predicteds - targets)
            return np.sqrt(np.cumsum(np.mean(squared_diff.reshape(len(predicteds), -1), axis=1)) /
                           np.arange(1, len(predicteds) + 1))

        rmse1 = compute_rmse(pred1, data1['targets'])
        rmse2 = compute_rmse(pred2, data2['targets'])

        print(f"\nRMSE 曲线比较:")
        print(f"  文件 1: 初始={rmse1[0]:.2e}, 最终={rmse1[-1]:.2e}")
        print(f"  文件 2: 初始={rmse2[0]:.2e}, 最终={rmse2[-1]:.2e}")
        print(f"  差异：初始={abs(rmse1[0]-rmse2[0]):.2e}, 最终={abs(rmse1[-1]-rmse2[-1]):.2e}")

    return all_match, "\n".join(messages)


def compare_files(file1: str, file2: str, tol: float = 1e-5, verbose: bool = False) -> bool:
    """比较两个 rollout 结果文件"""
    print(f"\n比较文件:")
    print(f"  文件 1: {file1}")
    print(f"  文件 2: {file2}")

    try:
        data1 = load_rollout_pkl(file1)
        data2 = load_rollout_pkl(file2)
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        return False

    match, msg = compare_rollout_results(data1, data2, tol, verbose)
    print(f"\n{msg}")

    return match


def compare_directories(dir1: str, dir2: str, tol: float = 1e-5, verbose: bool = False) -> bool:
    """比较两个目录中的所有 rollout 结果文件"""
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    if not dir1.exists():
        print(f"❌ 目录不存在：{dir1}")
        return False
    if not dir2.exists():
        print(f"❌ 目录不存在：{dir2}")
        return False

    # 查找两个目录中的所有 result*.pkl 文件（排除 ._ 开头的元数据文件）
    def find_result_files(d):
        files = {}
        for f in d.glob("result*.pkl"):
            if not f.name.startswith('._'):
                # 提取结果索引
                import re
                match = re.match(r'result(\d+)\.pkl', f.name)
                if match:
                    idx = int(match.group(1))
                    files[idx] = f
        return files

    files1 = find_result_files(dir1)
    files2 = find_result_files(dir2)

    common_indices = set(files1.keys()) & set(files2.keys())
    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())

    print(f"\n目录比较:")
    print(f"  目录 1: {dir1} ({len(files1)} 个结果文件)")
    print(f"  目录 2: {dir2} ({len(files2)} 个结果文件)")
    print(f"  共同结果：{len(common_indices)} 条轨迹")
    if only_in_1:
        print(f"  仅在目录 1: 轨迹 {sorted(only_in_1)}")
    if only_in_2:
        print(f"  仅在目录 2: 轨迹 {sorted(only_in_2)}")

    # 比较共同文件
    all_match = True
    results = {}

    for idx in sorted(common_indices):
        file1 = files1[idx]
        file2 = files2[idx]
        print(f"\n{'='*60}")
        print(f"比较轨迹 {idx}:")
        print(f"  {file1}")
        print(f"  {file2}")

        try:
            data1 = load_rollout_pkl(str(file1))
            data2 = load_rollout_pkl(str(file2))
            match, msg = compare_rollout_results(data1, data2, tol, verbose)
            results[idx] = match
            all_match = all_match and match
        except Exception as e:
            results[idx] = False
            print(f"❌ 比较失败：{e}")
            all_match = False

    # 总结
    print(f"\n{'='*60}")
    print("比较总结:")
    for idx, match in sorted(results.items()):
        status = "✅" if match else "❌"
        print(f"  {status} 轨迹 {idx}")

    if only_in_1 or only_in_2:
        print("\n⚠️  未匹配的轨迹:")
        for idx in sorted(only_in_1):
            print(f"  仅在目录 1: 轨迹 {idx}")
        for idx in sorted(only_in_2):
            print(f"  仅在目录 2: 轨迹 {idx}")

    print(f"\n{'='*60}")
    if all_match and not only_in_1 and not only_in_2:
        print("✅ 所有轨迹结果一致！")
        return True
    else:
        print("❌ 部分轨迹结果不一致或缺失")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='MeshGraphNet Rollout 结果比较工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 比较两个结果文件
  python compare_rollout_results.py outputs/v1/results/result0.pkl outputs/v2/results/result0.pkl

  # 比较两个目录
  python compare_rollout_results.py outputs/batch_rollout/xxx/results/ outputs/batch_rollout_v2/xxx/results/

  # 指定容差
  python compare_rollout_results.py file1.pkl file2.pkl --tol 1e-4

  # 详细输出
  python compare_rollout_results.py file1.pkl file2.pkl --verbose
        """
    )

    parser.add_argument('path1', help='第一个文件或目录路径')
    parser.add_argument('path2', help='第二个文件或目录路径')
    parser.add_argument('--tol', type=float, default=1e-5, help='数值容差 (默认：1e-5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    path1 = Path(args.path1)
    path2 = Path(args.path2)

    # 判断是文件还是目录
    if path1.is_file() and path2.is_file():
        success = compare_files(str(path1), str(path2), args.tol, args.verbose)
    elif path1.is_dir() and path2.is_dir():
        success = compare_directories(str(path1), str(path2), args.tol, args.verbose)
    else:
        print("❌ 路径类型不匹配：请提供两个文件或两个目录")
        if path1.exists() and not path2.exists():
            print(f"❌ 路径 2 不存在：{path2}")
        elif path2.exists() and not path1.exists():
            print(f"❌ 路径 1 不存在：{path1}")
        success = False

    exit(0 if success else 1)


if __name__ == '__main__':
    main()
