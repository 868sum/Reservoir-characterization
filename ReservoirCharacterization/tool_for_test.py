import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tool_for_pre import inverse_normalize_and_load
def plot_results(true_values, predicted_values, save_path, depth, args):
    """
    分别绘制折线图和散点图，并分别保存。
    """

    # 反归一化
    depth, true_values = inverse_normalize_and_load(
        np.array(depth), true_values,
        scaler_dir=os.path.join('data_save', '本次数据读取的缓存', args.input_directory),
        args=args
    )
    _, predicted_values = inverse_normalize_and_load(
        np.array(depth), predicted_values,
        scaler_dir=os.path.join('data_save', '本次数据读取的缓存', args.input_directory),
        args=args
    )

    depth = np.array(depth)
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # 设置大字体
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.titlesize'] = 26
    mpl.rcParams['axes.labelsize'] = 22
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['legend.fontsize'] = 20

    # =====================
    # 第一张图：折线图 Line Plot
    # =====================
    fig, ax = plt.subplots(figsize=(4, 11))  # 高度拉长

    ax.plot(true_values, depth, color='blue', linewidth=4, label='True Values')
    ax.plot(predicted_values, depth, color='red', linewidth=3, label='Predicted Values')

    ax.invert_yaxis()
    ax.set_xlabel(args.predict_target, labelpad=10)  # X轴名字
    ax.set_ylabel('Depth (m)', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # 把 legend 加到整个 figure 上方，单独放
    # 不是 ax.legend()，而是 fig.legend()
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.6, 1.03),  # 1.04位置稍微低一点，自己可以继续调
        ncol=1,  # 一列两行
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 给主图留出上面5%的空间给legend，不压图本身

    # 保存折线图
    line_save_path = save_path.replace('.png', '_line.png')
    plt.savefig(line_save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ 保存折线图: {line_save_path}")

    # =====================
    # 第二张图：散点图 Scatter Plot
    # =====================
    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values, alpha=0.7, edgecolors='k', s=60, c='green', marker='o')

    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('True Values', labelpad=10)
    plt.ylabel('Predicted Values', labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    scatter_save_path = save_path.replace('.png', '_scatter.png')
    plt.savefig(scatter_save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ 保存散点图: {scatter_save_path}")


def print_log(content, arg, log_file_path="output/"):
    # 确保日志文件夹存在
    log_file_path2 = log_file_path + arg.model_name + "_log.txt"
    os.makedirs(os.path.dirname(log_file_path2), exist_ok=True)
    # 打开文件并写入内容
    with open(log_file_path2, 'a') as log_file:
        log_file.write(content + "\n")
