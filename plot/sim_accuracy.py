import matplotlib.pyplot as plt
import numpy as np
import json

# 假设数据
predicted = np.random.rand(1000) * 100  # 预估性能
actual = predicted + np.random.randn(1000) * 5  # 实际性能（加噪声模拟）


def plot(source_file, DIFF, save_file, TITLE):
    FONT_SIZE = 40
    figsize = {
        # "figure.figsize": (9.2,6),  # Column, Row
        "figure.figsize": (6.6,6),  # Column, Row
        'font.sans-serif': 'Times New Roman',
        'axes.labelsize': FONT_SIZE,
        'font.size':FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    plt.rcParams.update(figsize)
    
    predicted = []
    actual = []
    with open(source_file, 'r') as f:
      intra_bsa_exe_plans_profile = json.load(f)
    for key, v in intra_bsa_exe_plans_profile.items():
        if 'sim_time' in v.keys():
            predicted.append(float(v['sim_time']))
            actual.append(float(v['time']))
    v_min = min(min(predicted), min(actual))
    v_max = max(max(predicted), max(actual))
    BASE = 2
    num_outliers = sum([0 if abs(x - y) / y < DIFF else 1 for x, y in zip(predicted, actual)])
    print(f'Outliers percentage: {num_outliers / len(predicted) * 100}%')
    # 创建散点图
    plt.figure()
    # plt.scatter(actual, predicted, alpha=0.3, label='Time Points', s=100)
    plt.scatter(actual, predicted, alpha=0.3, s=20)
    # plt.plot([v_min, v_max], [v_min, v_max], 'k-', label='y=x (Perfect Prediction)')  # 参考线
    plt.plot([v_min, v_max], [v_min * (1 - DIFF), v_max * (1 - DIFF)], 'r--', lw=2)  # 参考线
    plt.plot([v_min, v_max], [v_min * (1 + DIFF), v_max * (1 + DIFF)], 'r--', lw=2)  # 参考线
    # plt.plot([0, 100], [0, 100], 'r--', label='y=x (Perfect Prediction)')  # 参考线
    plt.xlabel('Actual Time(s)')
    plt.ylabel('Predicted Time(s)')
    plt.xscale('log', base=BASE)
    plt.yscale('log', base=BASE)
    plt.subplots_adjust(bottom=0.2, left=0.15)
    # plt.title(TITLE)
    # plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    plt.savefig(save_file)
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)            # Residual sum of squares
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)   # Total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    print(f"R² score: {r2:.4f}")
    
    
def main():
    plot('./database_bsa_infer/hamming/H100/intra_bsa_exe_plans_profile.json', 0.3, f'./plot/figs/sim_intra.pdf', 'Intra-node Distributed Attention')
    plot('./database_bsa_train/hamming/H100/inter_bsa_exe_plans_profile.json', 0.5, f'./plot/figs/sim_inter.pdf', 'Inter-node Distributed Attention')
    # [TODO]: contain dense_train
    
if __name__ == '__main__':
    main()
