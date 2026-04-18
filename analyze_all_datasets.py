"""
对比分析所有 TFRecord 数据集
"""

import tensorflow as tf
import numpy as np
import os

# 启用 eager execution
tf.compat.v1.enable_eager_execution()

DATA_DIR = "/Volumes/seagate/300Learn/310CodePractice/bytime/0417_meshgraphnet_understand/meshGraphNets_pytorch/data"

def analyze_dataset(split):
    """分析一个数据集"""
    file_path = os.path.join(DATA_DIR, f"{split}.tfrecord")

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return None

    # 统计记录数
    count = 0
    for _ in tf.data.TFRecordDataset(file_path):
        count += 1

    # 读取第一条记录进行详细分析
    dataset = tf.data.TFRecordDataset(file_path).take(1)
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # 解码数据
        mesh_pos = np.frombuffer(features['mesh_pos'].bytes_list.value[0], dtype=np.float32)
        node_type = np.frombuffer(features['node_type'].bytes_list.value[0], dtype=np.int32)
        cells = np.frombuffer(features['cells'].bytes_list.value[0], dtype=np.int32)
        velocity = np.frombuffer(features['velocity'].bytes_list.value[0], dtype=np.float32)

        # 计算形状
        N = mesh_pos.shape[0] // 2
        T = velocity.shape[0] // (N * 2)
        num_cells = cells.shape[0] // 3

        # 速度统计
        vel_reshaped = velocity.reshape(T, N, 2)

        # 节点类型统计
        type_counts = {}
        for t in np.unique(node_type):
            type_counts[int(t)] = int(np.sum(node_type == t))

        return {
            'split': split,
            'num_trajectories': count,
            'num_nodes': N,
            'num_cells': num_cells,
            'time_steps': T,
            'velocity_min': float(vel_reshaped.min()),
            'velocity_max': float(vel_reshaped.max()),
            'node_types': type_counts,
            'file_size_gb': os.path.getsize(file_path) / (1024**3)
        }

    return None


def print_table(results):
    """打印对比表格"""
    print("\n" + "="*90)
    print(" " * 30 + "数据集对比分析")
    print("="*90)

    # 表头
    print(f"\n{'数据集':<12} {'轨迹数':<10} {'节点数':<10} {'单元数':<10} {'时间步':<10} {'文件大小':<12}")
    print("-"*90)

    for r in results:
        if r:
            print(f"{r['split']:<12} {r['num_trajectories']:<10} {r['num_nodes']:<10} "
                  f"{r['num_cells']:<10} {r['time_steps']:<10} {r['file_size_gb']:.2f} GB")

    print("\n" + "="*90)
    print(" " * 35 + "节点类型分布")
    print("="*90)

    type_names = {0: 'NORMAL', 4: 'INFLOW', 5: 'OUTFLOW', 6: 'WALL'}

    for r in results:
        if r:
            print(f"\n{r['split'].upper()}:")
            for type_id, count in r['node_types'].items():
                name = type_names.get(type_id, f'UNKNOWN_{type_id}')
                pct = count / r['num_nodes'] * 100
                print(f"  {name} ({type_id}): {count} 节点 ({pct:.1f}%)")

    print("\n" + "="*90)
    print(" " * 35 + "速度场统计")
    print("="*90)

    for r in results:
        if r:
            print(f"\n{r['split'].upper()}:")
            print(f"  速度范围：[{r['velocity_min']:.4f}, {r['velocity_max']:.4f}]")


if __name__ == "__main__":
    print("正在分析数据集，请稍候...\n")

    results = []
    for split in ['train', 'valid', 'test']:
        print(f"分析 {split}.tfrecord ...")
        result = analyze_dataset(split)
        if result:
            results.append(result)

    print_table(results)

    # 总结
    print("\n" + "="*90)
    print(" " * 35 + "总结")
    print("="*90)

    total_trajectories = sum(r['num_trajectories'] for r in results)
    total_size = sum(r['file_size_gb'] for r in results)

    print(f"\n总轨迹数：{total_trajectories}")
    print(f"总文件大小：{total_size:.2f} GB")
    print(f"\n每条轨迹数据量:")
    if results:
        r = results[0]
        print(f"  网格节点：{r['num_nodes']}")
        print(f"  三角单元：{r['num_cells']}")
        print(f"  时间步数：{r['time_steps']}")
        print(f"  速度场数据：{r['time_steps'] * r['num_nodes'] * 2:,} 个 float32 值")
