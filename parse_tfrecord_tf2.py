#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse TFRecord - TensorFlow 数据集格式转换脚本 (TF2 + meta.json 版)

本脚本负责将 DeepMind 原始 TFRecord 格式的数据集转换为
PyTorch 可用的格式 (.npz + .dat)。

依赖 data/meta.json 文件定义数据格式

使用方法:
    python parse_tfrecord_tf2.py
"""

import tensorflow as tf
import functools
import json
import os
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# 启用 eager execution (TF2 默认启用，但为了兼容性)
tf.compat.v1.enable_eager_execution()


def load_meta(path):
    """加载 meta.json 并构建解析配置"""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())

    # 构建 field_names 列表
    meta['field_names'] = list(meta['features'].keys())

    # 推断 trajectory_length
    for key, field in meta['features'].items():
        if field['type'] == 'dynamic' and 'shape' in field:
            # 如 velocity: [600, -1, 2]
            shape = field['shape']
            if len(shape) >= 1 and shape[0] > 0:
                meta['trajectory_length'] = shape[0]
                break

    if 'trajectory_length' not in meta:
        meta['trajectory_length'] = 600  # 默认值

    print(f"Meta 信息:")
    print(f"  - 字段：{meta['field_names']}")
    print(f"  - 轨迹长度：{meta['trajectory_length']}")

    return meta


def _parse(proto, meta):
    """
    解析单个 TFRecord 样本
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
        if key not in features:
            continue

        # 解码原始字节数据
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))

        # 处理形状（-1 表示需要推断）
        shape = field.get('shape', [-1])
        # 将 -1 替换为实际值
        if -1 in shape:
            # 需要根据数据推断
            pass
        data = tf.reshape(data, shape)

        # 根据字段类型处理数据
        if field['type'] == 'static':
            # 静态字段：在时间维度上复制 (轨迹长度份)
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            # 变长字段：需要根据长度信息重建 RaggedTensor
            if 'length_' + key in features:
                length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
                length = tf.reshape(length, [-1])
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        # 动态字段保持不变

        out[key] = data

    return out


def load_dataset(path, split, meta):
    """
    加载 TFRecord 数据集
    """
    # 创建 TFRecord 数据集
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))

    # 使用 TF2 的 map
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)

    # 预取优化
    ds = ds.prefetch(1)

    return ds


def process_dataset(path, split, meta):
    """
    处理单个数据集划分
    """
    print(f"\n{'='*60}")
    print(f"处理 {split} 数据集...")
    print(f"{'='*60}")

    ds = load_dataset(path, split, meta)

    # 用于累积所有轨迹的数据
    all_pos = []
    all_node_type = []
    all_velocity = []
    all_cells = []

    # 输出文件路径
    filename = os.path.join(path, split + '.dat')

    # ==================== 第一遍遍历：计算总形状 ====================
    print("第一遍遍历：计算数据总形状...")
    shape0, shape1 = 0, 0
    num_trajectories = 0

    for index, d in enumerate(ds):
        velocity = d['velocity'].numpy()
        # 转置：[T, N, D] → [N, T, D]
        velocity = velocity.transpose(1, 0, 2)
        N, T, D = velocity.shape
        shape0 += N  # 累计节点数
        shape1 = max(shape1, T)  # 最大时间步数
        num_trajectories += 1

        if (index + 1) % 20 == 0:
            print(f"  已处理 {index + 1} 条轨迹...")

        del velocity

    print(f"  总节点数：{shape0}")
    print(f"  时间步数：{shape1}")
    print(f"  轨迹数量：{num_trajectories}")

    # ==================== 创建 memmap 文件 ====================
    print(f"\n创建 memmap 文件：{filename}")
    print(f"  形状：({shape0}, {shape1}, 2)")
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(shape0, shape1, 2))

    # ==================== 第二遍遍历：写入数据 ====================
    print("\n第二遍遍历：写入数据...")
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

        if (index + 1) % 20 == 0:
            print(f"  已写入 {index + 1} 条轨迹...")

    del fp

    # ==================== 构建索引数组 ====================
    print("\n构建索引数组...")
    # indices: 每条轨迹的节点起始位置
    indices = [i.shape[0] for i in all_pos]
    indices = np.cumsum(indices)
    indices = np.insert(indices, 0, 0)

    # cindices: 每条轨迹的网格单元起始位置
    cindices = [i.shape[0] for i in all_cells]
    cindices = np.cumsum(cindices)
    cindices = np.insert(cindices, 0, 0)

    # 拼接所有数据
    print("拼接数据...")
    all_pos = np.concatenate(all_pos, axis=0)
    all_node_type = np.concatenate(all_node_type, axis=0)
    all_cells = np.concatenate(all_cells, axis=0)

    # ==================== 保存为 .npz 文件 ====================
    npz_path = os.path.join(path, split + '.npz')
    print(f"保存 .npz 文件：{npz_path}")
    np.savez_compressed(
        npz_path,
        pos=all_pos,
        node_type=all_node_type,
        cells=all_cells,
        indices=indices,
        cindices=cindices,
        all_velocity_shape=(shape0, shape1, 2)
    )

    # 清理内存
    del all_pos, all_node_type, all_cells, all_velocity

    print(f"\n{split} 数据集处理完成!")
    print(f"  - 总节点数：{indices[-1]}")
    print(f"  - 轨迹数：{len(indices) - 1}")
    print(f"  - 速度场形状：{(shape0, shape1, 2)}")


if __name__ == '__main__':
    # ==================== 配置 ====================
    tf_dataset_path = 'data'

    # 检查数据目录
    if not os.path.exists(tf_dataset_path):
        print(f"错误：数据目录 '{tf_dataset_path}' 不存在")
        exit(1)

    # 检查 meta.json
    meta_path = os.path.join(tf_dataset_path, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"错误：未找到 meta.json 文件")
        print("请下载 meta.json 到 data 目录")
        exit(1)

    # 加载元数据
    meta = load_meta(tf_dataset_path)

    # ==================== 处理每个数据集划分 ====================
    for split in ['test', 'valid', 'train']:
        tfrecord_path = os.path.join(tf_dataset_path, split + '.tfrecord')
        if os.path.exists(tfrecord_path):
            process_dataset(tf_dataset_path, split, meta)
        else:
            print(f"\n跳过 {split}: 未找到 {tfrecord_path}")

    print("\n" + "="*60)
    print("所有数据处理完成!")
    print("="*60)
