"""
Parse TFRecord - TensorFlow 数据集格式转换脚本

本脚本负责将 DeepMind 原始 TFRecord 格式的数据集转换为
PyTorch 可用的格式 (.npz + .dat)。

为什么需要转换？
──────────────────────────────────────────────────────────────
DeepMind 的 MeshGraphNets 原始数据使用 TensorFlow 的 TFRecord
格式存储，而本项目是 PyTorch 实现，需要转换为 NumPy 格式。

输入数据 (DeepMind):
┌─────────────────────────────────────────────────────────────┐
│ cylinder_flow/                                              │
│ ├── train.tfrecord  # 训练轨迹                              │
│ ├── valid.tfrecord  # 验证轨迹                              │
│ ├── test.tfrecord   # 测试轨迹                              │
│ └── meta.json       # 数据格式描述                          │
└─────────────────────────────────────────────────────────────┘

输出数据 (本项目):
┌─────────────────────────────────────────────────────────────┐
│ data/                                                       │
│ ├── train.npz       # 元数据 (位置、类型、网格、索引)       │
│ ├── train.dat       # 速度场 (memmap 格式)                   │
│ ├── valid.npz                                             │
│ ├── valid.dat                                             │
│ ├── test.npz                                              │
│ └── test.dat                                              │
└─────────────────────────────────────────────────────────────┘

使用方法:
    # 1. 下载原始数据
    aria2c https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
    aria2c https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
    aria2c https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data

    # 2. 运行转换脚本
    python parse_tfrecord.py

注意:
    需要 TensorFlow < 1.15 (仅用于数据解析，不参与训练)
"""

import tensorflow as tf
import functools
import json
import os
import numpy as np
from packaging import version

# 版本检查：必须使用 TensorFlow < 1.15
# 原因：TFRecord 格式在新版本中有所变化
if version.parse(tf.__version__) >= version.parse("1.15"):
    raise RuntimeError(
        f"当前 TensorFlow 版本为 {tf.__version__}，但本项目要求 tensorflow<1.15。"
        "请在其他环境安装 TensorFlow：pip install 'tensorflow<1.15'"
    )


def _parse(proto, meta):
    """
    解析单个 TFRecord 样本

    Args:
        proto: TFRecord 序列化数据
        meta: 元数据 (从 meta.json 加载)

    Returns:
        out: 解析后的字典
            - mesh_pos: 网格节点位置
            - node_type: 节点类型
            - velocity: 速度场
            - cells: 网格单元 (三角形)

    字段类型:
        - static: 静态字段 (不随时间变化)，如 mesh_pos, node_type, cells
        - dynamic: 动态字段 (随时间变化)，如 velocity
        - dynamic_varlen: 变长动态字段
    """
    # 构建特征解析配置
    feature_lists = {
        k: tf.io.VarLenFeature(tf.string)
        for k in meta['field_names']
    }

    # 解析 TFExample
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}

    for key, field in meta['features'].items():
        # 解码原始字节数据
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])

        # 根据字段类型处理数据
        if field['type'] == 'static':
            # 静态字段：在时间维度上复制 (轨迹长度份)
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            # 变长字段：需要根据长度信息重建 RaggedTensor
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')

        out[key] = data

    return out


def load_dataset(path, split):
    """
    加载 TFRecord 数据集

    Args:
        path: 数据目录
        split: 数据集划分 ('train', 'valid', 'test')

    Returns:
        ds: TensorFlow Dataset 对象

    数据加载流程:
    1. 读取 meta.json 获取数据格式描述
    2. 创建 TFRecordDataset
    3. 使用 _parse 函数映射每条记录
    4. .prefetch(1) 预取优化
    """
    # 读取元数据
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())

    # 创建 TFRecord 数据集
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))

    # 并行解析 (num_parallel_calls=1 表示顺序解析)
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)

    # 预取优化
    ds = ds.prefetch(1)

    return ds


if __name__ == '__main__':
    # ==================== 配置 ====================
    tf_datasetPath = 'data'

    # 启用 TensorFlow 资源变量和 eager 执行模式
    tf.enable_resource_variables()
    tf.enable_eager_execution()

    # ==================== 处理每个数据集划分 ====================
    for split in ['train', 'test', 'valid']:
        print(f"\n处理 {split} 数据集...")

        ds = load_dataset(tf_datasetPath, split)

        # 用于累积所有轨迹的数据
        all_pos = []
        all_node_type = []
        all_velocity = []
        all_cells = []

        # 输出文件路径
        filename = os.path.join(tf_datasetPath, split + '.dat')

        # ==================== 第一遍遍历：计算总形状 ====================
        # 为什么要先遍历一遍？为了预分配 memmap 的精确大小
        shape0, shape1 = 0, 0
        for index, d in enumerate(ds):
            velocity = d['velocity'].numpy()
            # 转置：[T, N, D] → [N, T, D]
            # 原因：后续访问模式是 [节点，时间步，特征]
            velocity = velocity.transpose(1, 0, 2)
            N, T, D = velocity.shape
            shape0 += N  # 累计节点数
            shape1 = max(shape1, T)  # 最大时间步数
            del velocity

        # ==================== 创建 memmap 文件 ====================
        # memmap 优势：数据在磁盘上，按需加载，不占内存
        # shape: [总节点数，时间步数，2] (2 表示 vx, vy)
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(shape0, shape1, 2))

        # ==================== 第二遍遍历：写入数据 ====================
        write_shift = 0
        for index, d in enumerate(ds):
            # 提取数据
            pos_ = d['mesh_pos'].numpy()
            node_type_ = d['node_type'].numpy()
            velocity = d['velocity'].numpy()
            cells_ = d['cells'].numpy()

            # 静态数据只取第一个时间步 (所有时间步相同)
            pos = pos_[0].copy()           # [N, 2]
            node_type = node_type_[0].copy()  # [N, 1]
            cells = cells_[0].copy()       # [num_cells, 3]

            # 及时删除大对象，释放内存
            del pos_
            del node_type_
            del cells_

            print(pos.shape, node_type.shape, velocity.shape, cells.shape)

            # 累积到列表
            all_pos.append(pos)
            all_node_type.append(node_type)
            all_cells.append(cells)

            # 转置速度数据并写入 memmap
            velocity = velocity.transpose(1, 0, 2)
            fp[write_shift:write_shift + velocity.shape[0]] = velocity

            # 刷新到磁盘
            fp.flush()
            write_shift += velocity.shape[0]
            del velocity
        del fp

        # ==================== 构建索引数组 ====================
        # indices: 每条轨迹的节点起始位置
        indices = [i.shape[0] for i in all_pos]
        indices = np.cumsum(indices)
        indices = np.insert(indices, 0, 0)

        # cindices: 每条轨迹的网格单元起始位置
        cindices = [i.shape[0] for i in all_cells]
        cindices = np.cumsum(cindices)
        cindices = np.insert(cindices, 0, 0)

        # 拼接所有数据
        all_pos = np.concatenate(all_pos, axis=0)
        all_node_type = np.concatenate(all_node_type, axis=0)
        all_cells = np.concatenate(all_cells, axis=0)

        # ==================== 保存为 .npz 文件 ====================
        np.savez_compressed(
            os.path.join(tf_datasetPath, split + '.npz'),
            pos=all_pos,
            node_type=all_node_type,
            cells=all_cells,
            indices=indices,
            cindices=cindices,
            all_velocity_shape=(shape0, shape1, 2)
        )

        print(f"{split} 数据集处理完成!")
        print(f"  - 节点总数：{all_pos.shape[0]}")
        print(f"  - 轨迹数量：{len(indices) - 1}")
        print(f"  - 速度场形状：{(shape0, shape1, 2)}")
