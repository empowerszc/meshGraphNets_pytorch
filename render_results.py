"""
Render Results - 结果可视化渲染脚本

本脚本负责将 rollout 推理的结果可视化为视频文件，
便于直观观察模型预测质量。

功能:
1. 读取 rollout 生成的.pkl 结果文件
2. 使用 matplotlib 绘制速度场云图
3. 多进程并行渲染帧图像
4. 使用 OpenCV 合成 MP4 视频

输出示例:
    videos/output0.mp4  # 第一条轨迹的可视化视频
    videos/output1.mp4  # 第二条轨迹的可视化视频

使用方法:
    python render_results.py

视频规格:
    - 分辨率：1700 x 800 像素
    - 帧率：20 FPS
    - 格式：XVID 编码的 AVI
    - 内容：上下两帧对比 (上：真值，下：预测)
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import glob
import matplotlib.tri as tri
from PIL import Image
import matplotlib.pyplot as plt


def fig2data(fig):
    """
    将 Matplotlib 图表转换为 numpy 数组

    这是渲染流程的核心工具函数，用于将绘制的图表
    转换为 OpenCV 可以处理的图像格式。

    Args:
        fig: Matplotlib Figure 对象

    Returns:
        image: [H, W, 4] RGBA 格式的 numpy 数组

    转换步骤:
    1. 绘制图表到 canvas
    2. 获取 canvas 的 ARGB 缓冲区
    3. 转换为 RGBA 格式 (roll 通道)
    4. 使用 PIL 转换为 numpy 数组
    """
    # 绘制图表
    fig.canvas.draw()

    # 获取画布尺寸
    w, h = fig.canvas.get_width_height()

    # 从 canvas 获取原始字节数据 (ARGB 格式)
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # ARGB → RGBA：将 Alpha 通道从第一位移到最后一位
    buf = np.roll(buf, 3, axis=2)

    # 使用 PIL 转换并转为 numpy 数组
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


def render(args):
    """
    渲染单帧图像

    Args:
        args: 元组 (i, result, crds, triang, v_max, v_min)
            - i: 帧索引 (跳过采样后的索引)
            - result: [predicteds, targets] 结果列表
            - crds: 节点坐标 [N, 2]
            - triang: 三角网格剖分对象
            - v_max, v_min: 速度范数的最大最小值 (用于颜色映射)

    Returns:
        img: 渲染的图像 [H, W, 3] RGB 格式
        i: 帧索引 (用于排序)

    渲染内容:
    ┌─────────────────────────────────────────────────────────┐
    │  Figure (17, 8)                                         │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Target                                            │  │
    │  │  Time @ 0.00 s                                     │  │
    │  │  [速度场云图 + 网格叠加]                            │  │
    │  └───────────────────────────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Prediction                                        │  │
    │  │  Time @ 0.00 s                                     │  │
    │  │  [速度场云图 + 网格叠加]                            │  │
    │  └───────────────────────────────────────────────────┘  │
    │               Colorbar (共用)                           │
    └─────────────────────────────────────────────────────────┘
    """
    skip = 5          # 跳帧：每 5 帧渲染一帧
    step = i * skip   # 实际时间步索引

    # 提取当前帧的目标和预测速度场
    target = result[1][step]      # [N, 2] 真值速度
    predicted = result[0][step]   # [N, 2] 预测速度

    # 创建双子图 (上下排列)
    fig, axes = plt.subplots(2, 1, figsize=(17, 8))

    # 计算速度范数 (标量场，用于云图显示)
    target_v = np.linalg.norm(target, axis=-1)       # ||v|| = sqrt(vx² + vy²)
    predicted_v = np.linalg.norm(predicted, axis=-1)

    # 清空坐标轴 (为绘制做准备)
    for ax in axes:
        ax.cla()
        # 绘制三角网格线 (黑色细线)
        ax.triplot(triang, 'o-', color='k', ms=0.5, lw=0.3)

    # 绘制速度场云图 (使用相同的颜色范围便于对比)
    handle1 = axes[0].tripcolor(triang, target_v, vmax=v_max, vmin=v_min)
    axes[1].tripcolor(triang, predicted_v, vmax=v_max, vmin=v_min)

    # 设置标题 (显示时间，假设时间步长 0.01s)
    axes[0].set_title('Target\nTime @ %.2f s' % (step * 0.01))
    axes[1].set_title('Prediction\nTime @ %.2f s' % (step * 0.01))

    # 添加共用颜色条
    fig.colorbar(handle1, ax=[axes[0], axes[1]])

    # 转换为 numpy 数组并移除 Alpha 通道
    img = fig2data(fig)[:, :, :3]
    plt.close(fig)  # 关闭图表，释放内存

    return img, i


if __name__ == '__main__':
    # ==================== 查找结果文件 ====================
    result_files = glob.glob('result/*.pkl')
    os.makedirs('videos', exist_ok=True)

    # ==================== 逐个处理每个轨迹结果 ====================
    for index, file in enumerate(result_files):
        print(f"\n处理文件：{file}")

        # 加载 rollout 结果
        with open(file, 'rb') as f:
            result, crds = pickle.load(f)

        # 构建三角网格剖分 (用于 matplotlib 绘图)
        triang = tri.Triangulation(crds[:, 0], crds[:, 1])

        # 视频输出配置
        file_name = 'videos/output%d.mp4' % index
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 编码
        out = cv2.VideoWriter(file_name, fourcc, 20.0, (1700, 800))

        # 计算速度范围 (用于颜色映射一致性)
        r_t = result[1]  # targets
        v_max = np.max(r_t)
        v_min = np.min(r_t)

        # ==================== 多进程并行渲染 ====================
        # 使用进程池并行渲染每帧图像
        # 渲染 600/5 = 120 帧 (因为 skip=5)
        with ProcessPoolExecutor() as executor:
            # 提交所有渲染任务
            futures = {
                executor.submit(
                    render, (i, result, crds, triang, v_max, v_min)
                ): i for i in range(600 // 5)
            }

            # 等待完成并写入视频
            for future in tqdm(as_completed(futures), total=len(futures)):
                img, i = future.result()
                # 调整图像大小以匹配视频分辨率
                img_resized = cv2.resize(img, (1700, 800))
                out.write(img_resized)

        # 释放视频写入器
        out.release()
        print('视频 %s 已保存' % file_name)
