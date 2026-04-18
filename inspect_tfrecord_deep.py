"""
深度解析 TFRecord 文件 - 查看实际数据形状和内容
"""

import tensorflow as tf
import numpy as np
import os

# 启用 eager execution
tf.compat.v1.enable_eager_execution()

DATA_DIR = "/Volumes/seagate/300Learn/310CodePractice/bytime/0417_meshgraphnet_understand/meshGraphNets_pytorch/data"

def deep_inspect_tfrecord(file_path):
    """深度检查 TFRecord 文件"""

    print(f"\n{'='*70}")
    print(f"深度检查：{file_path}")
    print(f"{'='*70}\n")

    # 读取第一条记录
    dataset = tf.data.TFRecordDataset(file_path).take(1)

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        features = example.features.feature

        print("解码每个字段:\n")

        # 解码 mesh_pos
        mesh_pos_bytes = features['mesh_pos'].bytes_list.value[0]
        mesh_pos = np.frombuffer(mesh_pos_bytes, dtype=np.float32)
        print(f"mesh_pos (原始): {mesh_pos.shape[0]} 个 float32 值")
        print(f"mesh_pos 前 10 个值：{mesh_pos[:10]}")

        # 解码 node_type
        node_type_bytes = features['node_type'].bytes_list.value[0]
        node_type = np.frombuffer(node_type_bytes, dtype=np.int32)
        print(f"\nnode_type (原始): {node_type.shape[0]} 个 int32 值")
        print(f"node_type 唯一值：{np.unique(node_type)}")
        print(f"node_type 前 10 个值：{node_type[:10]}")

        # 解码 cells
        cells_bytes = features['cells'].bytes_list.value[0]
        cells = np.frombuffer(cells_bytes, dtype=np.int32)
        print(f"\ncells (原始): {cells.shape[0]} 个 int32 值")
        print(f"cells 前 10 个值：{cells[:10]}")

        # 解码 velocity
        velocity_bytes = features['velocity'].bytes_list.value[0]
        velocity = np.frombuffer(velocity_bytes, dtype=np.float32)
        print(f"\nvelocity (原始): {velocity.shape[0]} 个 float32 值")

        # 解码 pressure
        pressure_bytes = features['pressure'].bytes_list.value[0]
        pressure = np.frombuffer(pressure_bytes, dtype=np.float32)
        print(f"\npressure (原始): {pressure.shape[0]} 个 float32 值")

        # 推断形状
        print("\n" + "="*50)
        print("形状推断:")
        print("="*50)

        N = mesh_pos.shape[0] // 2  # 假设 2D 网格
        print(f"\n网格节点数 N ≈ {N}")

        # velocity 形状推断
        # 假设 velocity 是 [T, N, 2] 或 [N, T, 2]
        vel_total = velocity.shape[0]
        print(f"velocity 总元素数：{vel_total}")

        # 尝试不同的形状组合
        if vel_total % (N * 2) == 0:
            T = vel_total // (N * 2)
            print(f"推断时间步数 T = {T}")
            print(f"可能的形状：velocity = [T={T}, N={N}, 2]")
            vel_reshaped = velocity.reshape(T, N, 2)
            print(f"  验证：{vel_reshaped.shape}")
            print(f"  速度范围：[{vel_reshaped.min():.4f}, {vel_reshaped.max():.4f}]")

        # cells 形状推断
        # cells 应该是 [num_cells, 3]
        if cells.shape[0] % 3 == 0:
            num_cells = cells.shape[0] // 3
            print(f"\n三角单元数 = {num_cells}")
            cells_reshaped = cells.reshape(num_cells, 3)
            print(f"cells 形状：[{num_cells}, 3]")


if __name__ == "__main__":
    deep_inspect_tfrecord(os.path.join(DATA_DIR, "valid.tfrecord"))
