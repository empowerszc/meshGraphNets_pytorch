"""
Normalization - 在线归一化模块

本模块实现了一个在线统计累积的归一化器，用于对模型输入和输出进行标准化处理。

为什么需要归一化？
──────────────────────────────────────────────────────────────
1. 数值稳定：防止特征尺度差异过大导致梯度爆炸/消失
2. 加速收敛：归一化后的损失函数等高线更接近圆形
3. 统一量纲：不同物理量 (速度、压力等) 可能有不同量纲

为什么使用在线累积？
──────────────────────────────────────────────────────────────
传统方法：预先扫描整个数据集计算均值和标准差
  - 缺点：需要额外遍历数据集，内存开销大

在线累积：在训练过程中逐步累积统计量
  - 优点：无需预扫描，内存友好，自适应数据分布
"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """
    在线归一化器

    功能：
    1. 归一化：z = (x - mean) / std
    2. 反归一化：x = z * std + mean
    3. 在线累积统计量：在训练过程中逐步更新 mean 和 std

    统计量累积原理:
    ┌─────────────────────────────────────────────────────────┐
    │ 累积和：acc_sum = Σx                                    │
    │ 平方和：acc_sum_squared = Σx²                           │
    │ 计数：  acc_count = N                                   │
    │                                                         │
    │ 均值：mean = acc_sum / acc_count                        │
    │ 方差：variance = (acc_sum_squared / N) - mean²          │
    │ 标准差：std = sqrt(variance)                            │
    └─────────────────────────────────────────────────────────┘

    参数:
        size: 特征维度 (如 2 表示加速度，11 表示节点特征)
        max_accumulations: 最大累积次数 (默认 10^6)，防止精度问题
        std_epsilon: 标准差最小值，防止除零 (默认 1e-8)
        name: 归一化器名称 (用于调试)
        device: 运行设备

    属性:
        _acc_count: 累积的样本总数
        _num_accumulations: 累积操作次数
        _acc_sum: 累积和
        _acc_sum_squared: 累积平方和

    示例:
        >>> normalizer = Normalizer(size=2, device='cuda')
        >>> # 训练时：归一化并累积统计
        >>> normalized = normalizer(data, accumulate=True)
        >>> # 推理时：仅归一化，不累积
        >>> normalized = normalizer(data, accumulate=False)
        >>> # 反归一化
        >>> original = normalizer.inverse(normalized)
    """

    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8,
                 name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations  # 最大累积次数

        self._std_epsilon = std_epsilon  # 标准差下限

        # 使用 register_buffer 注册持久化张量
        # 这些张量会随着模型一起保存和加载
        self.register_buffer('_acc_count', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('_num_accumulations', torch.tensor(0.0, dtype=torch.float32))
        # 累积和 [1, size]
        self.register_buffer('_acc_sum', torch.zeros((1, size), dtype=torch.float32))
        # 累积平方和 [1, size]
        self.register_buffer('_acc_sum_squared', torch.zeros((1, size), dtype=torch.float32))
        self.to(device)

    def forward(self, batched_data, accumulate=True):
        """
        归一化前向传播

        Args:
            batched_data: 输入数据 [batch_size, size]
            accumulate: 是否累积统计量
                       - 训练时：True (更新统计量)
                       - 推理时：False (仅使用已累积的统计量)

        Returns:
            归一化后的数据 [batch_size, size]

        注意:
        - accumulate=True 时，仅在 _num_accumulations < _max_accumulations 时才累积
        - 这是为了防止累积次数过多导致数值精度问题
        """
        if accumulate:
            # 停止累积的条件：达到最大累积次数
            # 原因：超过 10^6 次后，浮点数精度可能出现问题
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())

        # 归一化公式：z = (x - μ) / σ
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """
        反归一化：将归一化数据还原为原始数据

        Args:
            normalized_batch_data: 归一化后的数据 [batch_size, size]

        Returns:
            原始数据 [batch_size, size]

        公式：x = z * σ + μ
        """
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """
        累积批次数据统计量

        这是 Welford 在线算法的简化版本：
        - 累加和用于计算均值
        - 累加平方和用于计算方差

        Args:
            batched_data: 当前批次数据 [batch_size, size]
        """
        count = batched_data.shape[0]  # 当前批次样本数
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)  # 当前批次和
        squared_data_sum = torch.sum(batched_data ** 2, dim=0, keepdim=True)  # 当前批次平方和

        # 更新累积量
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        """
        计算累积均值

        使用 safe_count = max(acc_count, 1) 避免除零
        在初始化阶段，acc_count=0，此时 mean=0
        """
        device = self._acc_count.device
        one_constant = torch.tensor(1.0, dtype=torch.float32, device=device)
        safe_count = torch.maximum(self._acc_count, one_constant)
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        """
        计算标准差并添加下限

        方差计算公式：variance = E[x²] - (E[x])²
        即：方差 = 平方和均值 - 均值平方

        数值稳定性处理:
        1. clamp(variance, min=0.0): 防止浮点误差导致负方差
        2. max(std, epsilon): 防止标准差为零导致除零

        Returns:
            标准差 [1, size]
        """
        device = self._acc_count.device
        one_constant = torch.tensor(1.0, dtype=torch.float32, device=device)
        safe_count = torch.maximum(self._acc_count, one_constant)

        mean = self._mean()
        # 方差 = E[x²] - (E[x])²
        variance = self._acc_sum_squared / safe_count - mean ** 2
        # 防止数值误差导致负方差
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        # 添加下限，防止除零
        std_epsilon = torch.tensor(self._std_epsilon, dtype=torch.float32, device=device)
        return torch.maximum(std, std_epsilon)
