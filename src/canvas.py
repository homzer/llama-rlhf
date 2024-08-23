import matplotlib.pyplot as plt
import numpy as np


def n_round(x: np.ndarray, n: int = 5):
    y = np.zeros_like(x)
    for i in range(1, x.shape[-1]):
        y[..., i] = np.mean(x[..., max(i - n, 0): i])
    return y


def draw_filled_curl(y: np.ndarray, x: np.ndarray = None):
    if x is None:
        x = np.arange(y.shape[0])

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', label='Curve')
    plt.fill_between(x, y, color='lightblue', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('title')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_histogram(count_dict: dict, x_label='X', y_label='Y', max_y=None, title=None):
    labels = []
    counts = []
    for item in count_dict.items():
        labels.append(item[0])
        counts.append(item[1])
    max_y_lim = max(counts) if max_y is None else max_y
    plt.figure()
    plt.bar(labels, counts, width=0.7, align='center')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, max_y_lim * 1.1)
    plt.show()


def draw_plots():
    x = np.linspace(2, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    y2 = np.cos(x) + np.random.normal(0, 0.1, x.shape)
    y3 = np.tan(x / 2) + np.random.normal(0, 0.1, x.shape)

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制三条折线
    plt.plot(x, y1, '-|', label='Sine')
    plt.plot(x, y2, '-|', label='Cosine')
    plt.plot(x, y3, '-|', label='Tangent')

    # 添加标题和坐标轴标签
    plt.title('GSM8K Test Set Performance', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xlabel('Training Epoch', fontsize=14)

    # 添加图例
    plt.legend(fontsize=12)

    # 设置网格
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 显示图表
    plt.show()


def draw_twins_plots():
    # 创建一些示例数据
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    y2 = np.cos(x) * 100 + np.random.normal(0, 1, x.shape)

    # 创建一个新的图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制第一个数据系列到 ax1（左边的 y 轴）
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Sin Value', color=color, fontsize=14)
    ax1.plot(x, y1, color=color, label='sin', linewidth=2, linestyle='-', marker='o', markersize=4, markevery=10)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    # 创建第二个 y 轴
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # 绘制第二个数据系列到 ax2（右边的 y 轴）
    color = 'tab:red'
    ax2.set_ylabel('Cos Value * 100', color=color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color, label='cos * 100', linewidth=2, linestyle='--', marker='^', markersize=4, markevery=10)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

    # 添加网格
    ax1.grid(True, which="both", ls="--", c=".5")

    # 设置图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 设置标题
    fig.suptitle('  ')
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2, fontsize='large')

    # 调整布局
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # 显示图形
    plt.show()


if __name__ == '__main__':
    draw_twins_plots()
