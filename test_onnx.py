"""
测试 ONNX 模型推理

验证导出的 ONNX 模型是否可以正常加载和推理
"""

import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("⚠️  未安装 onnxruntime，请运行：pip install onnxruntime")


def test_onnx_inference():
    """测试 ONNX 模型推理"""

    if not HAS_ORT:
        return

    print("="*70)
    print("ONNX 模型推理测试")
    print("="*70)

    # 加载模型
    print("\n加载模型...")
    try:
        sess = ort.InferenceSession('model.onnx')
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败：{e}")
        return

    # 获取输入信息
    print("\n输入信息:")
    for i, input_info in enumerate(sess.get_inputs()):
        print(f"  输入 {i}: {input_info.name}, 形状={input_info.shape}, 类型={input_info.type}")

    print("\n输出信息:")
    for i, output_info in enumerate(sess.get_outputs()):
        print(f"  输出 {i}: {output_info.name}, 形状={output_info.shape}, 类型={output_info.type}")

    # 创建测试输入
    print("\n创建测试输入...")
    num_nodes = 100
    num_edges = 300

    node_attr = np.random.randn(num_nodes, 11).astype(np.float32)
    edge_attr = np.random.randn(num_edges, 3).astype(np.float32)
    edge_index = np.random.randint(0, num_nodes, (2, num_edges), dtype=np.int64)

    print(f"  node_attr: {node_attr.shape}")
    print(f"  edge_attr: {edge_attr.shape}")
    print(f"  edge_index: {edge_index.shape}")

    # 运行推理
    print("\n运行推理...")
    outputs = sess.run(['acceleration'], {
        'node_attr': node_attr,
        'edge_attr': edge_attr,
        'edge_index': edge_index
    })

    acceleration = outputs[0]
    print(f"✓ 推理成功!")
    print(f"  输出形状：{acceleration.shape}")
    print(f"  输出范围：[{acceleration.min():.6f}, {acceleration.max():.6f}]")
    print(f"  输出均值：{acceleration.mean():.6f}")
    print(f"  输出标准差：{acceleration.std():.6f}")

    # 性能测试
    print("\n性能测试 (10 次推理平均)...")
    import time

    times = []
    for _ in range(10):
        start = time.time()
        sess.run(['acceleration'], {
            'node_attr': node_attr,
            'edge_attr': edge_attr,
            'edge_index': edge_index
        })
        times.append(time.time() - start)

    avg_time = np.mean(times)
    print(f"  平均推理时间：{avg_time*1000:.2f} ms")
    print(f"  标准差：{np.std(times)*1000:.2f} ms")

    print("\n" + "="*70)
    print("✓ 所有测试通过!")
    print("="*70)


def compare_pytorch_onnx():
    """比较 PyTorch 和 ONNX Runtime 的输出"""

    print("\n" + "="*70)
    print("PyTorch vs ONNX Runtime 输出对比")
    print("="*70)

    import torch
    from model.model import ONNXExportableMeshGraphNet

    # 创建 PyTorch 模型
    print("\n创建 PyTorch 模型...")
    torch_model = ONNXExportableMeshGraphNet(
        message_passing_num=15,
        node_input_size=11,
        edge_input_size=3
    )
    torch_model.eval()

    # 创建测试输入
    num_nodes = 100
    num_edges = 300

    node_attr = torch.randn(num_nodes, 11)
    edge_attr = torch.randn(num_edges, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # PyTorch 推理
    print("PyTorch 推理...")
    with torch.no_grad():
        torch_output = torch_model(node_attr, edge_attr, edge_index).numpy()

    # ONNX 推理
    print("ONNX 推理...")
    sess = ort.InferenceSession('model.onnx')
    onnx_output = sess.run(['acceleration'], {
        'node_attr': node_attr.numpy(),
        'edge_attr': edge_attr.numpy(),
        'edge_index': edge_index.numpy()
    })[0]

    # 比较
    print(f"\nPyTorch 输出：范围=[{torch_output.min():.6f}, {torch_output.max():.6f}]")
    print(f"ONNX 输出：  范围=[{onnx_output.min():.6f}, {onnx_output.max():.6f}]")

    diff = np.abs(torch_output - onnx_output).max()
    print(f"\n最大差异：{diff:.10f}")

    if diff < 1e-5:
        print("✓ 输出一致！(差异 < 1e-5)")
    elif diff < 1e-4:
        print("✓ 输出基本一致！(差异 < 1e-4，可能是数值精度差异)")
    else:
        print(f"⚠ 输出差异较大，请检查模型导出过程")

    print("="*70)


if __name__ == "__main__":
    test_onnx_inference()

    if HAS_ORT:
        try:
            compare_pytorch_onnx()
        except Exception as e:
            print(f"\n对比测试失败：{e}")
