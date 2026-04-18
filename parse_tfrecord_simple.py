#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse TFRecord - MeshGraphNets 数据格式转换脚本 (TF2 简化版)

直接从 TFRecord 解析数据，无需 meta.json 文件
支持 TensorFlow 2.x

使用方法:
    python parse_tfrecord_simple.py
"""

import tensorflow as tf
import os
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# 确保 eager execution 启用
tf.compat.v1.enable_eager_execution()


def decode_bytes(data_bytes, dtype_str, shape):
    """解码字节数据"""
    dtype_map = {
        'float32': tf.float32,
        'float64': tf.float64,
        'int32': tf.int32,
        'int64': tf.int64,
        'uint8': tf.uint8,
    }
    dtype = dtype_map.get(dtype_str, tf.float32)
    data = tf.io.decode_raw(data_bytes[0], dtype)
    data = tf.reshape(data, shape)
    return data.numpy()


def parse_record(raw_record):
    """解析单条 TFRecord 记录"""
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    data = {}

    # 解析 bytes 字段
    for key in ['mesh_pos', 'node_type', 'velocity', 'cells', 'pressure']:
        if key in example.features.feature:
            data[key] = example.features.feature[key].bytes_list.value

    return data


def try_decode_field(data, field_name, dtype_str, shape):
    """尝试解码字段"""
    if field_name not in data:
        return None
    try:
        return decode_bytes(data[field_name], dtype_str, shape)
    except Exception as e:
        print(f"解码 {field_name} 失败：{e}")
        return None


def process_tfrecord(tfrecord_path, output_dir, split):
    """处理单个 TFRecord 文件"""
    print(f"\n{'='*60}")
    print(f"处理 {split} 数据集: {tfrecord_path}")
    print(f"{'='*60}\n")

    # 统计记录数
    num_records = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    print(f"轨迹数量：{num_records}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 第一遍：读取所有轨迹，确定形状
    print("\n第一遍遍历：读取数据结构...")
    trajectories = []

    for idx, raw_record in enumerate(tf.data.TFRecordDataset(tfrecord_path)):
        data = parse_record(raw_record)

        # 尝试解码字段
        # mesh_pos: [N, 2] - 节点位置
        # node_type: [N, 1] - 节点类型
        # velocity: [T, N, 2] - 速度场
        # cells: [num_cells, 3] - 网格单元

        # 先从 mesh_pos 推断 N（最可靠）
        N = None
        if 'mesh_pos' in data:
            mesh_pos_flat = decode_bytes(data['mesh_pos'], 'float32', [-1])
            if len(mesh_pos_flat) % 2 == 0:
                N = len(mesh_pos_flat) // 2

        # 然后解码 velocity
        velocity = None
        T = None
        if 'velocity' in data and N is not None:
            velocity_flat = decode_bytes(data['velocity'], 'float32', [-1])
            total_size = len(velocity_flat)
            # velocity 原始形状是 [T, N, 2]
            expected_size = N * 2
            if total_size % expected_size == 0:
                T = total_size // expected_size
                velocity = velocity_flat.reshape(T, N, 2)
                print(f"  轨迹 {idx}: velocity shape = {velocity.shape} (N={N}, T={T})")
            else:
                print(f"  轨迹 {idx}: velocity 大小不匹配，total={total_size}, expected={expected_size}")

        if velocity is not None:
            # 解码其他字段
            mesh_pos = None
            if 'mesh_pos' in data:
                mesh_pos_flat = decode_bytes(data['mesh_pos'], 'float32', [-1])
                mesh_pos = mesh_pos_flat.reshape(N, 2)

            node_type = None
            if 'node_type' in data:
                node_type_flat = decode_bytes(data['node_type'], 'float32', [-1])
                if len(node_type_flat) == N:
                    node_type = node_type_flat.reshape(N, 1)
                else:
                    print(f"  轨迹 {idx}: node_type 形状不匹配，期望 {N}, 实际 {len(node_type_flat)}")

            cells = None
            if 'cells' in data:
                # cells 是 int 类型
                cells_flat = decode_bytes(data['cells'], 'int32', [-1])
                # 推断 num_cells
                if len(cells_flat) % 3 == 0:
                    num_cells = len(cells_flat) // 3
                    cells = cells_flat.reshape(num_cells, 3)
                else:
                    print(f"  轨迹 {idx}: cells 长度不是 3 的倍数：{len(cells_flat)}")

            trajectories.append({
                'mesh_pos': mesh_pos,
                'node_type': node_type,
                'velocity': velocity,
                'cells': cells
            })

        if (idx + 1) % 20 == 0:
            print(f"  已解析 {idx + 1}/{num_records} 条轨迹")

    if not trajectories:
        print("错误：未能解析任何有效的轨迹数据")
        return

    # 使用第一条轨迹的形状作为参考
    ref_traj = trajectories[0]
    N = ref_traj['mesh_pos'].shape[0]
    T = ref_traj['velocity'].shape[0]
    num_cells = ref_traj['cells'].shape[0]

    print(f"\n数据形状:")
    print(f"  - 节点数 N = {N}")
    print(f"  - 时间步 T = {T}")
    print(f"  - 网格单元数 = {num_cells}")
    print(f"  - 轨迹数 = {len(trajectories)}")

    # 第二遍：写入 memmap 文件
    print(f"\n第二遍遍历：写入数据文件...")

    # 计算总节点数（所有轨迹的节点数总和）
    total_nodes = sum(len(t['mesh_pos']) for t in trajectories)
    print(f"总节点数：{total_nodes}")

    fp = np.memmap(
        os.path.join(output_dir, split + '.dat'),
        dtype='float32',
        mode='w+',
        shape=(total_nodes, T, 2)
    )

    all_pos = []
    all_node_type = []
    all_cells = []
    indices = [0]
    cindices = [0]

    write_shift = 0
    for i, traj in enumerate(trajectories):
        N_i = traj['mesh_pos'].shape[0]
        T_i = traj['velocity'].shape[0]

        # 写入速度数据 (转置为 [N, T, 2])
        velocity_TND = traj['velocity']  # [T, N, 2]
        velocity_NTD = velocity_TND.transpose(1, 0, 2)  # [N, T, 2]
        fp[write_shift:write_shift + N_i] = velocity_NTD

        # 累积静态数据
        all_pos.append(traj['mesh_pos'])
        all_node_type.append(traj['node_type'])
        all_cells.append(traj['cells'])

        indices.append(write_shift + N_i)
        cindices.append((i + 1) * traj['cells'].shape[0])

        write_shift += N_i

        if (i + 1) % 20 == 0:
            print(f"  已写入 {i + 1}/{len(trajectories)} 条轨迹")

    fp.flush()
    del fp

    # 拼接并保存元数据
    print("\n保存元数据...")
    all_pos = np.concatenate(all_pos, axis=0)
    all_node_type = np.concatenate(all_node_type, axis=0)
    all_cells = np.concatenate(all_cells, axis=0)
    indices = np.array(indices)
    cindices = np.array(cindices)

    np.savez_compressed(
        os.path.join(output_dir, split + '.npz'),
        pos=all_pos,
        node_type=all_node_type,
        cells=all_cells,
        indices=indices,
        cindices=cindices,
        all_velocity_shape=(total_nodes, T, 2)
    )

    print(f"\n{split} 数据集处理完成!")
    print(f"  - 总节点数：{total_nodes}")
    print(f"  - 轨迹数：{len(trajectories)}")
    print(f"  - 输出文件：{split}.dat, {split}.npz")


if __name__ == '__main__':
    data_dir = 'data'

    if not os.path.exists(data_dir):
        print(f"错误：数据目录 '{data_dir}' 不存在")
        exit(1)

    # 处理每个数据集划分
    for split in ['test', 'valid', 'train']:
        tfrecord_path = os.path.join(data_dir, split + '.tfrecord')
        if os.path.exists(tfrecord_path):
            process_tfrecord(tfrecord_path, data_dir, split)
        else:
            print(f"\n跳过 {split}: 未找到 {tfrecord_path}")

    print("\n" + "="*60)
    print("所有数据处理完成!")
    print("="*60)
