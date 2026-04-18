"""
查看 TFRecord 文件内容的脚本

用于理解 DeepMind MeshGraphNets 数据集的格式
"""

import tensorflow as tf
import json
import os

# 启用 eager execution
tf.compat.v1.enable_eager_execution()

DATA_DIR = "/Volumes/seagate/300Learn/310CodePractice/bytime/0417_meshgraphnet_understand/meshGraphNets_pytorch/data"

def inspect_tfrecord(file_path, num_records=3):
    """
    检查 TFRecord 文件内容

    Args:
        file_path: TFRecord 文件路径
        num_records: 要检查的记录数量
    """
    print(f"\n{'='*70}")
    print(f"检查文件：{file_path}")
    print(f"{'='*70}\n")

    # 统计文件中的记录数
    count = 0
    for _ in tf.data.TFRecordDataset(file_path):
        count += 1
    print(f"总记录数 (轨迹数): {count}\n")

    # 读取前几条记录
    dataset = tf.data.TFRecordDataset(file_path).take(num_records)

    for i, raw_record in enumerate(dataset):
        print(f"--- 轨迹 {i} ---")

        # 解析为 Example 格式
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # 查看所有字段
        print("\n字段列表:")
        for key in example.features.feature.keys():
            print(f"  - {key}")

        # 查看每个字段的形状
        print("\n字段详情:")
        for key, value in example.features.feature.items():
            # 尝试解码不同的数据类型
            data = None
            if value.HasField('bytes_list'):
                data = value.bytes_list.value
                dtype = "bytes"
            elif value.HasField('float_list'):
                data = value.float_list.value
                dtype = "float"
            elif value.HasField('int64_list'):
                data = value.int64_list.value
                dtype = "int64"

            if data:
                if dtype == "bytes":
                    print(f"  {key}: {dtype}, len={len(data)}")
                else:
                    print(f"  {key}: {dtype}, len={len(data)}")

        print()


def decode_field(feature, field_name, dtype_bytes):
    """解码 TFRecord 字段"""
    data = tf.io.decode_raw(feature[field_name].values, dtype_bytes)
    return data


if __name__ == "__main__":
    # 检查 train 和 test 数据集
    for split in ['valid', 'test']:
        tfrecord_path = os.path.join(DATA_DIR, f"{split}.tfrecord")
        if os.path.exists(tfrecord_path):
            inspect_tfrecord(tfrecord_path, num_records=1)
        else:
            print(f"文件不存在：{tfrecord_path}")

    print("\n" + "="*70)
    print("总结:")
    print("="*70)
    print("""
TFRecord 文件结构说明:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每条记录 (Example) 代表一个完整的轨迹 (trajectory)，包含:

1. 静态字段 (不随时间变化):
   - mesh_pos: 网格节点位置 [N, 2]
   - node_type: 节点类型索引 [N, 1]
   - cells: 三角网格单元 [num_cells, 3]

2. 动态字段 (随时间变化):
   - velocity: 速度场 [T, N, 2]

3. 元数据字段:
   - length_velocity: 速度场的时间长度 T
   - 其他 field_names 列表

字段命名规则:
- bytes 类型存储原始数据
- 需要通过 meta.json 中的 dtype 和 shape 信息解码
""")
