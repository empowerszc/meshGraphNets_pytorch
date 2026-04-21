#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出随机初始化的模型（固定种子，可复现）
用于验证不同 batch rollout 实现的正确性
"""

import torch
import argparse
from model.simulator import Simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='checkpoints/test_random.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"随机种子：{args.seed}")

    # 创建模型
    model = Simulator(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3,
        device='cpu'
    )

    model.eval()

    # 保存
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'seed': args.seed
    }

    torch.save(checkpoint, args.output)
    print(f"模型已导出：{args.output}")
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")


if __name__ == '__main__':
    main()
