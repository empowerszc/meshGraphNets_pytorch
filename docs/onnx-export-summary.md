# ONNX 导出功能实现总结

> 时间：2026-04-17  
> 状态：✅ 完成

---

## 一、实现内容

### 1.1 新增文件

| 文件 | 功能 |
|------|------|
| `export_onnx.py` | ONNX 导出主脚本 |
| `test_onnx.py` | ONNX 模型测试脚本 |
| `docs/onnx-export-guide.md` | ONNX 导出详细指南 |

### 1.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `model/blocks.py` | 添加 ONNX 兼容类：`scatter_add_onnx`, `EdgeBlockONNX`, `NodeBlockONNX`, `GnBlockONNX` |
| `model/model.py` | 添加 `EncoderONNX`, `ONNXExportableMeshGraphNet` 类，为 `GnBlock` 添加 `forward_onnx` 方法 |
| `README.md` | 添加 ONNX 导出使用说明 |

---

## 二、使用方法

### 2.1 导出模型

```bash
# 导出随机权重模型（测试架构）
python export_onnx.py --output model.onnx

# 导出训练好的模型
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx --visualize
```

### 2.2 可视化

```bash
# 安装 Netron
pip install netron

# 打开浏览器可视化
netron model.onnx
```

### 2.3 测试推理

```bash
python test_onnx.py
```

---

## 三、技术要点

### 3.1 ONNX 导出挑战

MeshGraphNet 基于 PyTorch Geometric，导出 ONNX 面临以下挑战：

1. **torch_scatter 操作不可导出**
   - 解决方案：实现纯 PyTorch 版本的 `scatter_add_onnx()`

2. **PyG Data 对象不兼容**
   - 解决方案：创建使用张量接口的 `*ONNX` 类

3. **动态图结构**
   - 解决方案：显式传递 `num_nodes` 参数

### 3.2 核心代码

**scatter_add_onnx 实现**:

```python
def scatter_add_onnx(edge_attr, receivers_idx, num_nodes):
    """ONNX 兼容的 scatter_add"""
    result = torch.zeros(num_nodes, edge_attr.shape[1],
                         device=edge_attr.device, dtype=edge_attr.dtype)
    result.scatter_add_(0, receivers_idx.unsqueeze(1).expand_as(edge_attr), edge_attr)
    return result
```

**ONNX 模型架构**:

```python
class ONNXExportableMeshGraphNet(nn.Module):
    """可导出 ONNX 的 MeshGraphNet"""

    def __init__(self, message_passing_num=15, node_input_size=11,
                 edge_input_size=3, hidden_size=128):
        # 编码器
        self.node_encoder = build_mlp(11, 128, 128, lay_norm=True)
        self.edge_encoder = build_mlp(3, 128, 128, lay_norm=True)

        # 15 层 GnBlock (使用 ONNX 兼容版本)
        self.gn_blocks = nn.ModuleList([
            GnBlockONNX(128) for _ in range(message_passing_num)
        ])

        # 解码器
        self.decoder = build_mlp(128, 128, 2, lay_norm=False)

    def forward(self, node_attr, edge_attr, edge_index):
        N = node_attr.shape[0]
        node_feat = self.node_encoder(node_attr)
        edge_feat = self.edge_encoder(edge_attr)

        for block in self.gn_blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, edge_index, N)

        return self.decoder(node_feat)
```

---

## 四、测试结果

### 4.1 导出成功

```
✓ 导出成功!
✓ ONNX 验证通过!

ONNX 模型信息:
  IR 版本：7
  Opset 版本：13
  生产者：pytorch

算子统计:
  Gemm: 132  (全连接层)
  Relu: 99   (激活函数)
  Add: 108   (残差连接等)
  ScatterElements: 15 (消息聚合)
  ...
```

### 4.2 推理测试

```
ONNX 模型推理测试
=================
✓ 模型加载成功
✓ 推理成功!
  输出形状：(100, 2)
  平均推理时间：7.82 ms
```

---

## 五、模型规格

### 5.1 输入输出

| 名称 | 形状 | 说明 |
|------|------|------|
| `node_attr` | `[N, 11]` | 节点特征 (速度 2D + 节点类型 one-hot 9D) |
| `edge_attr` | `[E, 3]` | 边特征 (dx, dy, distance) |
| `edge_index` | `[2, E]` | 边索引 |
| `acceleration` | `[N, 2]` | 加速度预测 (ax, ay) |

### 5.2 参数量

- **隐空间维度**: 128
- **消息传递层数**: 15 层 GnBlock
- **总参数量**: ~2.5M
- **总算子数**: ~1,200

---

## 六、下一步

1. ✅ 导出功能完成
2. ✅ 测试脚本完成
3. ✅ 文档完成
4. ⏳ 使用训练好的 checkpoint 验证（需要训练模型）
5. ⏳ Netron 可视化（在浏览器中打开 model.onnx）

---

## 七、参考资料

- [ONNX 规范](https://onnx.ai/onnx/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Netron 可视化器](https://netron.app/)
- [PyTorch ONNX 导出](https://pytorch.org/docs/stable/onnx.html)
- [docs/onnx-export-guide.md](docs/onnx-export-guide.md)
