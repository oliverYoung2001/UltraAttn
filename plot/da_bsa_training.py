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

def plot_all_inter_configs(inter_exp_da_configs, prof_db: Prof_DB): # Relative Performance
    """
    inter_exp_da_configs: List[{'exp_config': xxx, 'da_config': xxx}]
    """
    with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'r') as f:
      inter_bsa_exe_plans_profile = json.load(f)
    
    sys_names = ['ring', 'w_node_tile', 'w_gpu_tile', 'w_kernel_tile', 'ultra']
    figsize = {
        "figure.figsize": (12,70),
        'font.sans-serif': 'Times New Roman',
        'axes.labelsize': 12,
        'font.size':8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    plt.rcParams.update(figsize)
    num_figs_per_row = 7
    num_figs = len(inter_exp_da_configs)
    fig, axs = plt.subplots(int(math.ceil(num_figs / num_figs_per_row)), num_figs_per_row)
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]

    # 用ABCDEF替代7个sys_name
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    abc = abc[: len(sys_names)]
    bar_width = 0.8
    bar_gap = 0.2
    # ylim = 1000 # 限制y轴范围, 保持表格整齐
    ylim = 1  # Upper bound of relative performance

    for c_id, exp_da_config in enumerate(inter_exp_da_configs):
        exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
        # Get times&performances !!!
        #   Create keys
        key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
        w_node_tile_suffix = f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)'
        key_suffixes = [w_node_tile_suffix]
        key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                            for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                                for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
        keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
        #   Parse and select execution times
        raw_time_dict = {key: float(inter_bsa_exe_plans_profile[key]['time']) for key in keys}
        ablation_time_dict = {
            'ring': float('inf'), # [TODO] @yqg
            'w_node_tile': raw_time_dict[f'{key_preffix}{w_node_tile_suffix}'],
            'w_gpu_tile': raw_time_dict[keys[1]],
            'w_kernel_tile': min(raw_time_dict[keys[1]], raw_time_dict[keys[2]]),
            'ultra': min([raw_time_dict[key] for key in keys[1:]]),
        }
        ablation_time_list = [ablation_time_dict[sys_name] for sys_name in sys_names]
        
        norm_perf = [ablation_time_list[-1] / t for t in ablation_time_list]
        # End
      
        fig_rid, fig_cid = c_id // num_figs_per_row, c_id % num_figs_per_row
        ax = axs[fig_rid, fig_cid]  # Get subfig
        
        # Raw Relative performance
        x_pos = np.arange(len(sys_names))
        bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

        # Special cases [TODO]
        
        # Text of speedup [TODO]
        
        # Labels of the subfig
        ax.set_xticks(range(len(abc)), abc)
        key_abbr = f'{"f" if exp_config.fob == 0 else "b"}_{da_config.bsa_config.CP}_S={da_config.shape_config["S"]}_Nh={da_config.shape_config["Nh"]}'
        ax.set_title(key_abbr, loc='center', fontsize=6)
        # if fig_rid == 0:
        #   ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

        # if fig_cid == 0:
          # ax.set_ylabel(devices[row_id], fontsize=10)
          # ax.yaxis.set_label_coords(-0.5, 0.5)
          # ax.yaxis.set_label_coords(0, 1)

        # max_height = np.nanmax(norm_perf)
        ax.set_ylim(0, ylim)
        ax.set_yticks(np.arange(0, ylim * 4 + 1, 1) / 4)  # [0, 0.25, 0.5, 0.75, 1]
     
    # Add legend to the global fig 
    legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
    fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.5,wspace=0.4)
    fig.savefig(f"./plot/figs/inter_bsa_configs_training.pdf", bbox_inches='tight')

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
    os.environ['CLUSTER_NAME'] = 'hamming'
    os.environ['PLATFORM'] = 'H100'
    prof_db = initialize_prof_db()
    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs()
    exp_configs = get_exp_configs()
    
    # Calc all inter_exp_da_configs
    inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    plot_all_inter_configs(inter_exp_da_configs, prof_db)
    
if __name__ == '__main__':
    main()