"""
FpcDataset - 圆柱绕流 (Flow Past Cylinder) 数据集加载模块

本模块负责从预处理后的数据文件中加载流场模拟数据，
并将其转换为 PyTorch Geometric 的 Data 对象供模型使用。

数据存储格式:
┌─────────────────────────────────────────────────────────────┐
│  .npz 文件 (元数据)                                         │
│  - pos:         网格节点坐标 [总节点数，2]                  │
│  - node_type:   节点类型 [总节点数，1]                      │
│  - cells:       网格单元 (三角形) [总三角形数，3]           │
│  - indices:     轨迹边界索引 [轨迹数 +1]                    │
│  - cindices:    网格单元边界索引 [轨迹数 +1]                │
│  - all_velocity_shape: 速度场数组形状                        │
├─────────────────────────────────────────────────────────────┤
│  .dat 文件 (速度场数据，memmap 格式)                        │
│  - 形状：[总样本数，时间步数，2]                            │
│  -  dtype: float32                                          │
│  - 内容：每个网格节点在每个时间步的速度 (vx, vy)            │
└─────────────────────────────────────────────────────────────┘

样本组织方式:
┌────────────────────────────────────────────────────────────┐
│ indices: [0, N₁, N₂, N₃, ...]                              │
│          │   │   │   │                                     │
│          │   │   │   └─ 轨迹 3 的起始索引                    │
│          │   │   └───── 轨迹 2 的起始索引                    │
│          │   └───────── 轨迹 1 的起始索引                    │
│          └───────────── 轨迹 0 的起始索引                    │
│                                                            │
│ 每个轨迹包含 T-1 个样本 (T=时间步数)                         │
│ 样本 i: (速度 t, 速度 t+1) 作为 (输入，目标) 对              │
└────────────────────────────────────────────────────────────┘
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FpcDataset(Dataset):
    """
    圆柱绕流数据集类

    功能:
    1. 从.npz 文件加载元数据 (网格位置、节点类型、三角面索引)
    2. 从.dat 文件 (memmap) 高效读取速度场数据
    3. 根据索引计算，返回指定轨迹的指定时间步的图数据

    参数:
        data_root: 数据目录路径
        split: 数据集划分 ('train', 'valid', 'test')

    属性:
        meta: 元数据字典
        fp: 速度场数据的 memmap 对象
        tra_len: 每条轨迹的时间步数
        num_sampes_per_tra: 每条轨迹的样本数 (= tra_len - 1)
        total_samples: 数据集总样本数

    示例:
        >>> dataset = FpcDataset(data_root='data', split='train')
        >>> len(dataset)  # 总样本数
        >>> graph = dataset[0]  # 获取第一个样本
        >>> graph.x  # 节点特征 [N, 11]
        >>> graph.y  # 目标速度 [N, 2]
        >>> graph.pos  # 节点位置 [N, 2]
        >>> graph.face  # 三角面 [3, num_faces]
    """

    def __init__(self, data_root, split):
        """
        初始化数据集

        Args:
            data_root: 数据目录路径
            split: 数据集划分 ('train', 'valid', 'test')
        """
        # 元数据文件路径 (.npz)
        meta_path = os.path.join(data_root, split + '.npz')
        # 速度场数据文件路径 (.dat)
        data_path = os.path.join(data_root, split + '.dat')
        # 需要加载的元数据键
        meta_keys = ("pos", "node_type", "cells", "indices", "cindices", "all_velocity_shape")

        # 加载元数据
        tmp = np.load(meta_path, allow_pickle=True)
        self.meta = {key: tmp[key] for key in meta_keys}

        # 使用 memmap 加载速度场数据
        # memmap 的优势：数据存储在磁盘上，按需读取，不占用大量内存
        shape = self.meta['all_velocity_shape']
        self.fp = np.memmap(data_path, dtype='float32', mode='r', shape=shape)

        # 计算轨迹相关参数
        self.tra_len = self.fp.shape[1]  # 时间步数 (如 1000)
        self.num_sampes_per_tra = self.tra_len - 1  # 每条轨迹的样本数 (999)
        tras_nums = len(self.meta['indices']) - 1  # 轨迹数量 (减去起始的 0 索引)
        self.total_samples = tras_nums * self.num_sampes_per_tra  # 总样本数

    def __getitem__(self, index):
        """
        根据索引获取单个样本

        索引计算逻辑:
        ┌─────────────────────────────────────────────────────────┐
        │ 假设 index = 1500, num_sampes_per_tra = 999            │
        │                                                         │
        │ tra_index = 1500 // 999 = 1        → 第 1 条轨迹         │
        │ tra_sample_index = 1500 % 999 = 501 → 该轨迹的第 501 步   │
        │                                                         │
        │ 轨迹 1 的节点范围：indices[1] ~ indices[2]                │
        │ 轨迹 1 的网格范围：cindices[1] ~ cindices[2]             │
        └─────────────────────────────────────────────────────────┘

        Args:
            index: 全局样本索引 (0 ~ total_samples-1)

        Returns:
            graph: PyTorch Geometric Data 对象
                - x: [N, 11] 节点特征 (node_type + velocity)
                - pos: [N, 2] 节点位置坐标
                - face: [3, num_faces] 三角面索引
                - y: [N, 2] 目标速度 (下一时刻)
        """
        # ========== 1. 计算轨迹索引和样本在轨迹内的索引 ==========
        tra_index = index // self.num_sampes_per_tra  # 属于第几条轨迹
        tra_sample_index = index % (self.tra_len - 1)  # 轨迹内的时间步索引

        # ========== 2. 获取该轨迹的边界索引 ==========
        # 节点索引范围
        tra_start_index = self.meta['indices'][tra_index]
        tra_end_index = self.meta['indices'][tra_index + 1]
        # 网格单元索引范围
        ctra_start_index = self.meta['cindices'][tra_index]
        ctra_end_index = self.meta['cindices'][tra_index + 1]

        # ========== 3. 提取速度场数据 ==========
        # 当前时刻速度 [N, 2]
        tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]
        # 下一时刻速度 (目标) [N, 2]
        tra_target = self.fp[tra_start_index:tra_end_index, tra_sample_index + 1]

        # ========== 4. 提取静态网格数据 ==========
        # 节点位置 [N, 2]
        pos = self.meta['pos'][tra_start_index:tra_end_index]
        # 节点类型 [N, 1]
        node_type = self.meta['node_type'][tra_start_index:tra_end_index]
        # 网格单元 (三角形) [num_faces, 3]
        cells = self.meta['cells'][ctra_start_index:ctra_end_index]

        # ========== 5. 构造节点特征 ==========
        # 拼接 node_type 和 velocity → [N, 1+2=3]
        # 注意：node_type 这里是标量索引，后续会被 one-hot 编码
        x = np.concatenate([node_type, tra_velocity], axis=-1)

        # ========== 6. 转换为 PyTorch 张量 ==========
        # .copy() 确保数据是可写的 (memmap 返回的是只读的)
        x = torch.as_tensor(x.copy(), dtype=torch.float32)
        pos = torch.as_tensor(pos.copy(), dtype=torch.float32)
        # face 需要转置为 [3, num_faces] 格式
        face = torch.as_tensor(cells.T.copy(), dtype=torch.int64)
        y = torch.as_tensor(tra_target.copy(), dtype=torch.float32)

        # ========== 7. 构造 PyG Data 对象 ==========
        graph = Data(x=x, pos=pos, face=face, y=y)

        return graph

    def __len__(self):
        """返回数据集总样本数"""
        return self.total_samples
