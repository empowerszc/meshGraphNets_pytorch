# MeshGraphNet ONNX 导出指南

> 导出模型为 ONNX 格式用于可视化和跨平台部署  
> 更新时间：2026-04-17

---

## 一、快速开始

### 1.1 安装依赖

```bash
conda activate meshgraphnet
pip install onnx netron
```

### 1.2 导出模型

```bash
# 导出随机权重的模型（仅测试架构）
python export_onnx.py --output model.onnx

# 导出训练好的模型
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx

# 导出并打印模型结构摘要
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx --visualize
```

### 1.3 可视化

**方法 1: 使用 Netron（推荐）**

```bash
# 安装 Netron
pip install netron

# 启动 Netron
netron model.onnx

# 或在浏览器中打开 https://netron.app/ 并上传 model.onnx
```

**方法 2: 使用 Python 检查**

```bash
python -c "import onnx; m = onnx.load('model.onnx'); print(m)"
```

---

## 二、ONNX 模型信息

### 2.1 输入输出规格

| 名称 | 形状 | 说明 |
|------|------|------|
| **node_attr** | `[N, 11]` | 节点特征（速度 2 + 节点类型 one-hot 9） |
| **edge_attr** | `[E, 3]` | 边特征（dx, dy, distance） |
| **edge_index** | `[2, E]` | 边索引（sender 索引，receiver 索引） |
| **acceleration** | `[N, 2]` | 输出：加速度预测（ax, ay） |

**符号说明**:
- N = 节点数（动态）
- E = 边数（动态）

### 2.2 模型架构统计

典型导出的 ONNX 模型包含：

| 算子类型 | 数量 | 说明 |
|----------|------|------|
| **Gemm** | 132 | 全连接层（Linear） |
| **Relu** | 99 | 激活函数 |
| **Add** | 108 | 逐元素加法（残差连接等） |
| **Concat** | 45 | 特征拼接 |
| **ReduceMean** | 64 | LayerNorm 中的均值计算 |
| **ScatterElements** | 15 | scatter_add 操作（消息聚合） |
| **Identity** | 193 | 权重占位符 |

**总层数**: ~1,200 层算子

### 2.3 模型参数

- **隐空间维度**: 128
- **消息传递层数**: 15 层 GnBlock
- **总参数量**: ~2.5M

---

## 三、技术细节

### 3.1 为什么需要 ONNX 导出？

| 优势 | 说明 |
|------|------|
| **可视化** | 通过 Netron 直观查看模型架构 |
| **调试** | 检查每一层的输出和中间张量 |
| **跨平台** | 可在 C++、JavaScript、移动端部署 |
| **优化** | 使用 ONNX Runtime 加速推理 |
| **互操作** | 与其他框架（TensorFlow、JAX）互转 |

### 3.2 导出挑战与解决方案

MeshGraphNet 基于 PyTorch Geometric，某些操作导出 ONNX 有限制：

| 挑战 | 解决方案 |
|------|----------|
| **torch_scatter.scatter_add** | 实现纯 PyTorch 版本 `scatter_add_onnx()` |
| **PyG Data 对象** | 使用张量接口而非 Data 对象 |
| **动态图结构** | 显式传递 num_nodes 参数 |
| **LayerNorm** | 使用 PyTorch 原生实现，可导出 |

### 3.3 ONNX 兼容的类层次

```
model/blocks.py:
├── EdgeBlockONNX      # ONNX 兼容的边更新模块
├── NodeBlockONNX      # ONNX 兼容的节点更新模块
└── GnBlockONNX        # ONNX 兼容的图网络块

model/model.py:
└── ONNXExportableMeshGraphNet  # 完整的可导出模型
```

### 3.4 scatter_add_onnx 实现

```python
def scatter_add_onnx(edge_attr, receivers_idx, num_nodes):
    """
    ONNX 兼容的 scatter_add 实现
    
    使用纯 PyTorch 操作：
    1. 创建零张量 [N, edge_dim]
    2. 使用 scatter_add_ 原地累加
    """
    result = torch.zeros(num_nodes, edge_attr.shape[1], 
                         device=edge_attr.device, dtype=edge_attr.dtype)
    result.scatter_add_(0, receivers_idx.unsqueeze(1).expand_as(edge_attr), edge_attr)
    return result
```

---

## 四、使用 ONNX 模型推理

### 4.1 使用 ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# 创建推理会话
sess = ort.InferenceSession('model.onnx')

# 准备输入
num_nodes = 100
num_edges = 300

node_attr = np.random.randn(num_nodes, 11).astype(np.float32)
edge_attr = np.random.randn(num_edges, 3).astype(np.float32)
edge_index = np.random.randint(0, num_nodes, (2, num_edges), dtype=np.int64)

# 推理
outputs = sess.run(['acceleration'], {
    'node_attr': node_attr,
    'edge_attr': edge_attr,
    'edge_index': edge_index
})

print(f"输出形状：{outputs[0].shape}")  # [100, 2]
```

### 4.2 使用 PyTorch 加载

```python
import torch
import onnx

# 验证 ONNX 模型
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# 使用 PyTorch 加载
import onnxruntime
ort_session = onnxruntime.InferenceSession('model.onnx')

# 或使用 torch.onnx 加载
torch_model = torch.onnx.load('model.onnx')
```

---

## 五、常见问题

### Q1: 导出时出现 TracerWarning

```
TracerWarning: Iterating over a tensor might cause the trace to be incorrect.
```

**解答**: 这是正常警告。`edge_index` 解包为 `senders_idx, receivers_idx` 时触发，但不影响导出结果。模型功能正常。

### Q2: 无法加载训练好的权重

```
Missing keys: ['node_encoder.0.weight', ...]
Unexpected keys: ['model.node_encoder.0.weight', ...]
```

**解答**: 权重 key 名称不匹配。训练时的 checkpoint 包含 `model.` 前缀和归一化器权重。导出脚本会自动处理：

```python
# 跳过归一化器权重
if 'normalizer' in key.lower():
    continue
# 移除前缀
new_key = key.replace('model.', '')
```

### Q3: ONNX 模型太大

**解答**: 使用动态轴（dynamic_axes）导出，支持不同节点/边数量：

```python
torch.onnx.export(
    model,
    inputs,
    output_path,
    dynamic_axes={
        'node_attr': {0: 'num_nodes'},
        'edge_attr': {0: 'num_edges'},
        'edge_index': {1: 'num_edges'},
        'acceleration': {0: 'num_nodes'}
    }
)
```

### Q4: 如何在 C++ 中使用？

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// 1. 创建 session
Ort::Session session(env, "model.onnx", session_options);

// 2. 准备输入张量
std::vector<int64_t> node_shape = {num_nodes, 11};
Ort::Value input_node = Ort::Value::CreateTensor<float>(
    memory_info, node_data.data(), node_size, node_shape.data(), 2
);

// 3. 运行推理
auto outputs = session.Run(
    Ort::RunOptions{nullptr}, 
    input_names, inputs.data(), num_inputs,
    output_names, num_outputs
);
```

---

## 六、性能优化

### 6.1 使用 ONNX Runtime 加速

```bash
# 安装 onnxruntime-gpu（如果需要 GPU）
pip install onnxruntime-gpu

# 启用 Graph Optimization
import onnxruntime as ort
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.optimized_model_filepath = "model_optimized.onnx"
sess = ort.InferenceSession('model.onnx', session_options)
```

### 6.2 量化加速

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# 量化为 INT8
quantize_dynamic('model.onnx', 'model_quantized.onnx', weight_type=QuantType.QUInt8)

# 量化后模型更小，推理更快（但精度略有损失）
```

---

## 七、下一步

1. **可视化模型**: 使用 Netron 查看完整架构
2. **验证输出**: 对比 PyTorch 和 ONNX Runtime 的输出
3. **性能测试**: 测量推理延迟和吞吐量
4. **部署应用**: 集成到 C++/Python/Web 应用

---

## 参考资源

- ONNX 规范：https://onnx.ai/onnx/
- ONNX Runtime: https://onnxruntime.ai/
- Netron 可视化器：https://netron.app/
- PyTorch ONNX 导出：https://pytorch.org/docs/stable/onnx.html
