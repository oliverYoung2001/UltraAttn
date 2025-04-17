import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from search_algo.initialize import initialize_prof_db
from search_algo.exp_configs import get_bsa_configs, get_exp_configs
from search_algo.search_engine import Dist_Attn_Config
import json
from typing import List
from search_algo.database import Prof_DB

def calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs):
    inter_node_shape_configs = shape_config_dict['inter']
    hierarchy = 0   # (0, 1) -> (inter, intra)
    inter_exp_da_configs: List[dict] = []
    for exp_config in exp_configs:
        for inter_node_bsa_config in inter_node_bsa_configs:
            for Nh in inter_node_shape_configs['Nhs']:
                for S_tot in inter_node_shape_configs['Ss']:
                    for bs in inter_node_shape_configs['BSs']:
                        for D in inter_node_shape_configs['Ds']:
                            shape_config = {
                                'Nh': (Nh, Nh),
                                'S': (S_tot, S_tot),  # S_tot
                                'bs': bs,
                                'D': D,
                            }
                            da_config = Dist_Attn_Config.from_bsa_config(
                                inter_node_bsa_config, 
                                shape_config=shape_config,
                                hierarchy=hierarchy,
                            )
                            S_per_gpu = S_tot // da_config.tot_sp
                            if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                                continue
                            inter_exp_da_configs.append({
                                'exp_config': exp_config,
                                'da_config': da_config,
                            })
    return inter_exp_da_configs

def plot_all_inter_configs(inter_exp_da_configs, prof_db: Prof_DB, fob: bool): # Relative Performance
    """
    inter_exp_da_configs: List[{'exp_config': xxx, 'da_config': xxx}]
    """
    # fob = 0
    CPs = [
      (8, 1),
      (8, 2),
      (8, 4),
      (8, 8),
    ]
    # Ss = [16384, 524288]  # 16K, 512K
    Ss = [65536, 524288]  # 64K, 512K
    # Ss = [131072, 524288]  # 128K, 512K
    Ss_str_dict = {
      '16384': '16k',
      '65536': '64k',
      '131072': '128k',
      '524288': '512k',
    }
    Nhs = [1, 32]
    bs = 1
    D = 128
    BSA_reprs = [
      '[[1110000000000000][1111000000000000][1111000000000000][0111000000000000][0000111000000000][0000111100000000][0000111100000000][0000011100000000][0000000011100000][0000000011110000][0000000011110000][0000000001110000][0000000000001110][0000000000001111][0000000000001111][0000000000000111]]',
      '[[1111111111111111][1100000000000000][1010000000000000][1001000000000000][1000100000000000][1000010000000000][1000001000000000][1000000100000000][1000000010000000][1000000001000000][1000000000100000][1000000000010000][1000000000001000][1000000000000100][1000000000000010][1000000000000001]]',
    ]
    BSA_NAMES = [
      'Strided',
      'Global+Local'
    ]
    # with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'r') as f:
    with open('./database_bsa_train/zhipu_hamming/H100/inter_bsa_exe_plans_profile.json', 'r') as f:
      inter_bsa_exe_plans_profile = json.load(f)
    FONT_SIZE = 18
    # sys_names = ['ring', 'w_node_tile', 'w_node+device_tile', 'w_node+device+kernel_tile', 'UltraAttn']
    sys_names = ['Ring', 'Node Tile', 'Node+Device Tile', 'Node+Device+Kernel Tile', 'UltraAttn']
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (20,2.8),  # Column, Row
        'font.sans-serif': 'Times New Roman',
        'axes.labelsize': FONT_SIZE,
        'font.size': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': 15,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    plt.rcParams.update(figsize)
    num_figs_per_row = len(CPs) * len(Ss) * len(Nhs)
    num_figs = num_figs_per_row * len(BSA_reprs)
    num_rows = num_figs // num_figs_per_row
    fig, axs = plt.subplots(num_rows, num_figs_per_row)
    # 调整子图之间的横向间隔
    # plt.subplots_adjust(wspace=0)  # 设置横向间隔为 0.1（默认值通常为 0.2）
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]
    hatch_def = [None] * len(sys_names)
    hatch_def = HATCH_DEF[:len(sys_names)-1] + [None]

    # 用ABCDEF替代7个sys_name
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    abc = abc[: len(sys_names)]
    bar_width = 0.8
    bar_gap = 0.2
    # bar_width = 0.4
    # bar_gap = 0.1
    # ylim = 1000 # 限制y轴范围, 保持表格整齐
    ylim = 1  # Upper bound of relative performance

    # for c_id, exp_da_config in enumerate(inter_exp_da_configs):
    for fig_rid, bsa_repr in enumerate(BSA_reprs):
        fig_cid = - 1
        for Nh in Nhs:
            for S in Ss:
                for CP in CPs:       
                    # exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
                    # Get times&performances !!!
                    #   Create keys
                    # key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
                    fig_cid += 1
                    shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                    bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                    key_preffix = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                    # sub_fig_title = f'CP={CP}\nS={Ss_str_dict[str(S)]},Nh={Nh}'
                    sub_fig_title = f'CP{math.prod(CP)}\nS={Ss_str_dict[str(S)]}\nNh={Nh}'
                    # End
                    
                    w_node_tile_suffix = f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)'
                    key_suffixes = ['_ring', w_node_tile_suffix]
                    key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                                        for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
                    keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
                    #   Parse and select execution times
                    raw_time_dict = {key: float(inter_bsa_exe_plans_profile[key]['time']) for key in keys}
                    ablation_time_dict = {  # 'Node Tile', 'Node+Device Tile', 'Node+Device+Kernel Tile'
                        'Ring': raw_time_dict[keys[0]], # @yqg
                        'Node Tile': raw_time_dict[f'{key_preffix}{w_node_tile_suffix}'],
                        # 'w_node+gpu_tile': raw_time_dict[keys[1]],
                        # 'w_node+gpu+kernel_tile': min(raw_time_dict[keys[1]], raw_time_dict[keys[2]]),
                        # 'ultra': min([raw_time_dict[key] for key in keys[1:]]),
                        'Node+Device Tile': raw_time_dict[keys[-4]],
                        'Node+Device+Kernel Tile': min(raw_time_dict[keys[-4]], raw_time_dict[keys[-3]]),
                        'UltraAttn': min([raw_time_dict[key] for key in keys[-4:]]),
                    }
                    ablation_time_list = [ablation_time_dict[sys_name] for sys_name in sys_names]
                    
                    norm_perf = [ablation_time_list[-1] / t for t in ablation_time_list]
                    # End
                  
                    ax = axs[fig_rid, fig_cid]  # Get subfig
                    
                    # Raw Relative performance
                    x_pos = np.arange(len(sys_names))
                    bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

                    # Special cases [TODO]
                    
                    # Text of speedup [TODO]
                    max_baseline = max(norm_perf[: 1])
                    # ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, - 0.15, f'TODO\u00D7', fontweight='bold', ha='center', va='bottom', \
                    #   fontsize=7, color='red')
                    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.2, 0.5, f'{norm_perf[-1]/max_baseline:.2f}\u00D7', fontweight='bold', ha='center', va='center', \
                      fontsize=FONT_SIZE, color='black', rotation=90)

                    # Labels of the subfig
                    # ax.set_xticks(range(len(abc)), abc)
                    ax.set_xticks([])
                    if fig_rid == num_rows - 1:
                        ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-0.9)
                    # if fig_rid == 0:
                    #   ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

                    if fig_cid == 0:
                      ax.set_ylabel(BSA_NAMES[fig_rid], fontdict={'weight': 'bold'})
                      ax.yaxis.set_label_coords(-0.5, 0.5)
                      # ax.yaxis.set_label_coords(0, 1)

                    ax.set_ylim(0, ylim)
                    if fig_cid == 0:
                        # ax.set_yticks(np.arange(0, ylim * 4 + 1, 1) / 4)  # [0, 0.25, 0.5, 0.75, 1]
                        ax.set_yticks(np.arange(0, ylim * 2 + 1, 1) / 2)  # [0, 0.5, 1]
                    else:
                        ax.set_yticks([])
     
    # Add legend to the global fig 
    # legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label=sys_names[i]) for i in range(len(sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.10))
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.05)
    fig.savefig(f"./plot/figs/inter_bsa_configs_training_cherry_pick_fob={fob}.pdf", bbox_inches='tight')

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

# plot([data_a100, data_h100], ['A100', 'H100'], model_names, sys_names, figure_name='e2e', add_legend=True)
 
  
def main():
    os.environ['CLUSTER_NAME'] = 'zhipu_hamming'
    os.environ['PLATFORM'] = 'H100'
    # prof_db = initialize_prof_db()
    # inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs()
    # exp_configs = get_exp_configs()
    
    # Calc all inter_exp_da_configs
    # inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    plot_all_inter_configs(None, None, fob=0)
    plot_all_inter_configs(None, None, fob=1)
    
if __name__ == '__main__':
    main()