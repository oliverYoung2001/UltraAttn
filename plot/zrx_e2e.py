from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

model_names = ['h2o', 'roco', 'keyformer', 'snapkv', 'corm', 'attn', 'gemma2']
sys_names = ['torch', 'dynamo', 'tensorrt', 'tvm', 'korch', 'einnet', 'our']

# data_a100 = parse_csv(LOG_DIR + "a100_40_e2e_4096.csv", sep='\t')
# data_h100 = parse_csv(LOG_DIR + "h100_80_e2e_4096.csv", sep='\t')

data_a100 = [
  [870.4972,	679.4771,	718.3669,	1032.9777,	1051.2273,	1656.8996,	306.9346],  # h2o
  [1012.6765,	698.7392,	733.9764,	1246.0357, 1206.9757,	2100.2619,	314.6816],  # roco
  [1554.342,	973.2994,	799.9457,	1054.7816,	-1,	5461.3712,	538.501], # keyformer
  [871.4715,	679.6971,	713.7223,	770.9506,	1049.7078,	1683.6359,	309.4149],  # snapkv
  [955.1325,	721.5211,	959.4448,	-1,	1278.8785,	-1,	330.5531],  # corm
  [826.3275,	659.1717,	371.3981,	613.6015,	995.6181,	1415.4351,	285.93],  # attn
  [968.8927,	720.3511,	414.6827,	641.8723,	995.1936,	1722.8929,	286.8143],  # gemma2
]
data_h100 = [
  [583.5969,	305.2832,	455.9139,	519.0936,	731.612,	790.2472,	189.2485],  # h2o
  [695.7338,	318.9485,	472.5503,	607.5242,	851.1218,	1018.086,	195.877],  # roco
  [1103.8482,	450.6336,	520.1073,	740.7938,	-1,	1422.3223,	319.9795], # keyformer
  [588.5703,	304.7682,	466.5934,	512.2075,	728.9676,	791.7283,	189.1349],  # snapkv
  [649.0136,	319.943,	640.0223,	-1,	785.513,	-1,	200.6378],  # corm
  [544.6888,	286.755,	199.491,	397.1777,	687.6937,	667.2986,	167.2563],  # attn
  [661.854,	304.9569,	254.9128,	409.0667,	689.9429,	852.743,	168.8982],  # gemma2
]

def plot(data, devices, model_names, sys_names, figure_name, add_legend=False):
  # 两图合并，参数有所修改
  figsize = {
      "figure.figsize": (12, 2),
      'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size':8,
      'legend.fontsize': 10,
      'xtick.labelsize': 10,
      'ytick.labelsize': 9,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
#   plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams.update(figsize)
  fig, axs = plt.subplots(2, len(model_names))

  # 和utils.py中的COLOR_DEF相同，共7种颜色
  pair_color_def = COLOR_DEF[:len(sys_names)]
  hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

  # 用ABCDEF替代7个sys_name
  abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  abc = abc[:len(sys_names)]
  bar_width = 0.8
  bar_gap = 0.2
  ylim = 1000 # 限制y轴范围, 保持表格整齐

  for row_id, (ax_row, dataset) in enumerate(zip(axs, data)):
    for col_id, (ax, model_name) in enumerate(zip(ax_row, model_names)):
      # perf_ref = dataset.loc[model_name][sys_names]
      perf_ref = dataset[col_id]

      # perf = perf_ref.clip(lower=0) 
      perf = [max(x, 0) for x in perf_ref]
      # 绝对时间
      # baseline = perf.loc[sys_names[-1]]  # scalar
      baseline = perf[-1]
      norm_perf = perf 

      x_pos = np.arange(len(sys_names))
      bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

      for i, bar in enumerate(bars):
        # if perf_ref.loc[sys_names[i]] == 0:
        if perf_ref[i] == 0:
          # OOM
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'OOM', ha='center', va='bottom', rotation=90)
        # elif perf_ref.loc[sys_names[i]] == -1:
        elif perf_ref[i] == -1:
          # 不支持
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'NS', ha='center', va='bottom', rotation=90)
        # elif perf_ref.loc[sys_names[i]] == -2:
        elif perf_ref[i] == -2:
          # 超时
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.05, 'TLE', ha='center', va='bottom', rotation=90)
        # elif norm_perf.loc[sys_names[i]] > ylim: 
        elif norm_perf[i] > ylim:
          # 截断，文字补充
          ax.text(bar.get_x() + bar.get_width() / 2, ylim * 0.95, f'{norm_perf[i]:.1f}', ha='center', va='top', rotation=90)
      
      min_perf = float('inf')
      for i in perf_ref[:-1]:
        if i > 0:
          min_perf = min(min_perf, i)
      # speedup
      ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, bars[-1].get_height(), f'{min_perf / baseline:.1f}\u00D7', fontweight='bold', ha='center', va='bottom', fontsize=7)

      ax.set_xticks(range(len(abc)), abc)
      # 子图标注model_name
      if row_id == 0:
        ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

      if col_id == 0:
        ax.set_ylabel(devices[row_id], fontsize=10)
        ax.yaxis.set_label_coords(-0.5, 0.5)

      max_height = np.nanmax(norm_perf)
      ax.set_ylim(0, ylim)
      ax.set_yticks(range(0, ylim + 1, 250))

  # 添加legend    
  legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + SYS_NAME[sys_names[i]]) for i in range(len(sys_names))]
  fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
  fig.text(0.085, 0.5, 'Execution Time (ms)', va='center', rotation='vertical', fontsize=10)
  plt.subplots_adjust(hspace=0.5,wspace=0.4)
  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')

plot([data_a100, data_h100], ['A100', 'H100'], model_names, sys_names, figure_name='e2e', add_legend=True)