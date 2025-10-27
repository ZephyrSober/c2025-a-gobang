import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def drawOnehot(tensor: torch.Tensor,force_plot= True) -> bool:
    # 转换为numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    class_indices = np.argmax(tensor, axis=-1)
    # 创建自定义颜色映射
    # 0: 透明, 1: 黑色, 2: 白色
    cmap = ListedColormap(['gray', 'black', 'white'])
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制图像，设置透明
    im = ax.imshow(class_indices, cmap=cmap, vmin=0, vmax=2)

    # 设置网格线以便更好地区分像素
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, 15, 1))
    ax.set_yticks(np.arange(-0.5, 15, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 移除坐标轴
    ax.tick_params(which='both', bottom=False, left=False)

    if(force_plot):
        plt.title('One-hot Tensor Visualization')
        plt.tight_layout()
        plt.show()

    return True

