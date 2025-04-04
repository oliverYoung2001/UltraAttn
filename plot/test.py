import matplotlib.pyplot as plt
import numpy as np

# 数据准备（模拟数据，基于图表中的大致数值）
models = ['(A) PyTorch', '(B) TorchInductor', '(C) TensorRT', '(D) TVM', '(E) Korch', '(F) EinNet', '(G) FlashTensor']
datasets = ['H2O', 'ROCo', 'Keyformer', 'SnapKV', 'Corm', 'Vanilla Attention', 'Gemm2']
colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99', '#ff99cc', '#99ccff', '#cc99ff']  # 对应 A-G 的颜色

# 模拟 A100 时间数据（单位：ms）
a100_data = {
    'H2O': [750, 500, 1033, 1051, 1656, 1030, 1229],
    'ROCo': [727, 500, 1246, 1270, 2011, 1011, 1277],
    'Keyformer': [555, 500, 1018, 1045, 1273, 1011, 1277],
    'SnapKV': [797, 500, 1047, 1085, 1273, 1011, 1277],
    'Corm': [797, 500, 1047, 1085, 1273, 1011, 1277],
    'Vanilla Attention': [415, 500, 1047, 1085, 1273, 1011, 1277],
    'Gemm2': [797, 500, 1047, 1085, 1273, 1011, 1277]
}

# 模拟 Execution 时间数据（单位：ms）
execution_data = {
    'H2O': [500, 250, 500, 500, 500, 500, 500],
    'ROCo': [500, 250, 500, 500, 500, 500, 500],
    'Keyformer': [500, 250, 500, 500, 500, 500, 500],
    'SnapKV': [500, 250, 500, 500, 500, 500, 500],
    'Corm': [500, 250, 500, 500, 500, 500, 500],
    'Vanilla Attention': [500, 250, 500, 500, 500, 500, 500],
    'Gemm2': [500, 250, 500, 500, 500, 500, 500]
}

# 倍数标注（模拟，基于图表中的标注）
multipliers = {
    'H2O': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'ROCo': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'Keyformer': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'SnapKV': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'Corm': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'Vanilla Attention': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×'],
    'Gemm2': [None, None, '2.2×', '2.2×', '2.2×', '2.2×', '2.2×']
}

# 创建画布和子图（2 行，7 列）
fig, axes = plt.subplots(2, 7, figsize=(20, 6), sharey='row')
fig.suptitle('(a) End-to-end performance', y=0.95)

# 绘制 A100 时间（第一行）
for i, dataset in enumerate(datasets):
    ax = axes[0, i]
    bars = ax.bar(models, a100_data[dataset], color=colors, edgecolor='black')
    ax.set_title(dataset)
    ax.set_xticks([])  # 隐藏 x 轴标签
    if i == 0:
        ax.set_ylabel('A100 Time (ms)')
    ax.set_ylim(0, 1000)

    # 添加倍数标注
    for bar, multiplier in zip(bars, multipliers[dataset]):
        if multiplier:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 20, multiplier, ha='center', color='red')

# 绘制 Execution 时间（第二行）
for i, dataset in enumerate(datasets):
    ax = axes[1, i]
    bars = ax.bar(models, execution_data[dataset], color=colors, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G'], rotation=0)
    if i == 0:
        ax.set_ylabel('Execution Time (ms)')
    ax.set_ylim(0, 1000)

    # 添加倍数标注
    for bar, multiplier in zip(bars, multipliers[dataset]):
        if multiplier:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 20, multiplier, ha='center', color='red')

# 添加图例
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black') for color in colors]
fig.legend(handles, models, loc='upper center', ncol=len(models), bbox_to_anchor=(0.5, 1.05))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 留出空间给图例
# plt.show()
plt.savefig(f'./plot/test0.png')