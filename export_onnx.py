"""
导出 MeshGraphNet 模型为 ONNX 格式

用于可视化模型结构和调试

使用方法:
    python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx

可视化 ONNX 模型:
    # 方法 1: 使用 Netron (推荐)
    pip install netron
    netron model.onnx  # 在浏览器中打开

    # 方法 2: 使用 onnxruntime 检查
    python -c "import onnx; onnx.load('model.onnx')"

注意:
    由于 PyTorch Geometric 的某些操作 (如 scatter_add) 导出 ONNX 有限制，
    本脚本创建了一个简化版本的模型用于可视化。
"""

import argparse
import os
import torch
import torch.nn as nn

# 导入项目模块
from model.model import ONNXExportableMeshGraphNet, build_mlp


def create_sample_inputs(num_nodes=100, num_edges=300):
    """
    创建示例输入用于 ONNX 导出

    Args:
        num_nodes: 节点数
        num_edges: 边数

    Returns:
        (node_attr, edge_attr, edge_index) 元组
    """
    # 节点特征 [N, 11] = [node_type_onehot(9), velocity(2)]
    node_attr = torch.randn(num_nodes, 11)

    # 边特征 [E, 3] = [dx, dy, distance]
    edge_attr = torch.randn(num_edges, 3)

    # 边索引 [2, E]
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    return node_attr, edge_attr, edge_index


def export_onnx(model, output_path, opset_version=13):
    """
    导出模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        output_path: 输出文件路径
        opset_version: ONNX opset 版本
    """
    # 创建示例输入
    num_nodes = 100
    num_edges = 300
    node_attr, edge_attr, edge_index = create_sample_inputs(num_nodes, num_edges)

    # 设置模型为评估模式
    model.eval()

    # 导出
    print(f"导出模型到：{output_path}")
    print(f"输入规格:")
    print(f"  - node_attr: {node_attr.shape}")
    print(f"  - edge_attr: {edge_attr.shape}")
    print(f"  - edge_index: {edge_index.shape}")

    torch.onnx.export(
        model,
        (node_attr, edge_attr, edge_index),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['node_attr', 'edge_attr', 'edge_index'],
        output_names=['acceleration'],
        dynamic_axes={
            'node_attr': {0: 'num_nodes'},
            'edge_attr': {0: 'num_edges'},
            'edge_index': {1: 'num_edges'},
            'acceleration': {0: 'num_nodes'}
        },
        verbose=False
    )

    print(f"✓ 导出成功!")

    # 验证 ONNX 文件
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX 验证通过!")

        # 打印模型信息
        print(f"\nONNX 模型信息:")
        print(f"  IR 版本：{onnx_model.ir_version}")
        print(f"  Opset 版本：{onnx_model.opset_import[0].version}")
        print(f"  生产者：{onnx_model.producer_name if onnx_model.producer_name else 'PyTorch'}")

        # 打印层数统计
        node_count = {}
        for node in onnx_model.graph.node:
            op_type = node.op_type
            node_count[op_type] = node_count.get(op_type, 0) + 1

        print(f"\n算子统计 (前 20 个):")
        for op_type, count in sorted(node_count.items(), key=lambda x: -x[1])[:20]:
            print(f"  {op_type}: {count}")

    except ImportError:
        print("⚠ 未安装 onnx，跳过验证 (pip install onnx)")
    except Exception as e:
        print(f"⚠ 验证失败：{e}")


def visualize_onnx_structure(onnx_path):
    """
    打印 ONNX 模型结构的可读摘要
    """
    try:
        import onnx
        model = onnx.load(onnx_path)

        print("\n" + "="*70)
        print("ONNX 模型结构摘要")
        print("="*70)

        # 输入
        print("\n输入:")
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")

        # 输出
        print("\n输出:")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")

        # 按层类型分组
        layers = {}
        for node in model.graph.node:
            op_type = node.op_type
            if op_type not in layers:
                layers[op_type] = []
            layers[op_type].append(node.name)

        print(f"\n层类型统计:")
        for op_type, names in sorted(layers.items()):
            print(f"  {op_type}: {len(names)} 层")

        # 详细层列表 (前 50 层)
        print(f"\n前 50 层详情:")
        for i, node in enumerate(model.graph.node[:50]):
            inputs = [inp for inp in node.input]
            outputs = [out for out in node.output]
            print(f"  [{i:3d}] {node.op_type:15s} -> {outputs[0] if outputs else 'N/A'}")

        if len(model.graph.node) > 50:
            print(f"  ... 还有 {len(model.graph.node) - 50} 层")

        print("\n" + "="*70)

    except ImportError:
        print("⚠ 未安装 onnx，无法显示结构摘要")


def main():
    parser = argparse.ArgumentParser(description='导出 MeshGraphNet 到 ONNX')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='训练好的 checkpoint 路径 (可选)')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='ONNX 输出文件路径')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset 版本 (默认 13)')
    parser.add_argument('--no_weights', action='store_true',
                        help='仅导出架构，不加载训练权重')
    parser.add_argument('--visualize', action='store_true',
                        help='导出后打印模型结构摘要')

    args = parser.parse_args()

    print("="*70)
    print("MeshGraphNet ONNX 导出工具")
    print("="*70)

    # 创建模型
    print("\n创建模型...")
    model = ONNXExportableMeshGraphNet(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3
    )

    # 加载训练权重 (如果提供)
    if args.checkpoint and not args.no_weights:
        if os.path.exists(args.checkpoint):
            print(f"加载 checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

            # 提取模型权重
            state_dict = checkpoint['model_state_dict']

            # 由于简化版本结构略有不同，需要过滤和转换权重
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # 跳过归一化器权重
                if 'normalizer' in key.lower():
                    continue
                # 转换 key 名称
                new_key = key.replace('model.', '')
                filtered_state_dict[new_key] = value

            # 加载权重
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            if missing:
                print(f"  缺少的权重：{missing}")
            if unexpected:
                print(f"  多余的权重：{unexpected}")

            print("✓ 权重加载完成")
        else:
            print(f"⚠️  checkpoint 不存在：{args.checkpoint}，使用随机权重")

    # 导出
    export_onnx(model, args.output, args.opset)

    # 可视化
    if args.visualize:
        visualize_onnx_structure(args.output)

    # 使用建议
    print("\n" + "="*70)
    print("下一步操作:")
    print("="*70)
    print("""
1. 使用 Netron 可视化:
   pip install netron
   netron {}

   或在浏览器打开：https://netron.app/ 并上传该文件

2. 使用 Python 检查:
   python -c "import onnx; m = onnx.load('{}'); print(m)"

3. 使用 onnxruntime 推理:
   python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('{}')
inputs = sess.get_inputs()
print('输入:', [i.name for i in inputs])
"
    """.format(args.output, args.output, args.output))


if __name__ == '__main__':
    main()
