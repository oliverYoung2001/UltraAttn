import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from search_algo.initialize import initialize_prof_db
from search_algo.exp_configs import get_bsa_infer_configs, get_exp_infer_configs
from search_algo.search_engine import Dist_Attn_Config
import json
from typing import List
from search_algo.database import Prof_DB

def calc_all_intra_exp_da_configs(intra_node_bsa_configs, shape_config_dict, exp_configs):
    intra_node_shape_configs = shape_config_dict['intra']
    hierarchy = 1   # (0, 1) -> (inter, intra)
    intra_exp_da_configs: List[dict] = []
    for exp_config in exp_configs:
      for intra_node_bsa_config in intra_node_bsa_configs:
            for Nh in intra_node_shape_configs['Nhs']:
                for S_per_node in intra_node_shape_configs['Ss']:
                    for bs in intra_node_shape_configs['BSs']:
                        for D in intra_node_shape_configs['Ds']:
                            shape_config = {
                                'Nh': (Nh, Nh),
                                'S': (S_per_node, S_per_node),    # Useless
                                'bs': bs,
                                'D': D,
                            }
                            da_config = Dist_Attn_Config.from_bsa_config(
                                intra_node_bsa_config, 
                                shape_config=shape_config,
                                hierarchy=hierarchy,
                            )
                            S_per_gpu = S_per_node // da_config.hierarchy_sp
                            if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                                continue
                            intra_exp_da_configs.append({
                                'exp_config': exp_config,
                                'da_config': da_config,
                            })
    return intra_exp_da_configs

def plot_all_intra_configs(intra_exp_da_configs, prof_db: Prof_DB = None): # Relative Performance
    """
    intra_exp_da_configs: List[{'exp_config': xxx, 'da_config': xxx}]
    """
    # with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
    #   intra_bsa_exe_plans_profile = json.load(f)
    with open('./database_bsa_infer/zhipu_hamming/H100/intra_bsa_exe_plans_profile.json', 'r') as f:
      intra_bsa_exe_plans_profile = json.load(f)
      
    # sys_names = ['ring', 'w_node_tile', 'w_gpu_tile', 'w_kernel_tile', 'ultra']
    sys_names = ['ring', 'w_node_tile', 'w_gpu_tile', 'w_gpu+kernel_tile', 'ultra']
    figsize = {
        "figure.figsize": (12,30),
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
    num_figs = len(intra_exp_da_configs)
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

    for c_id, exp_da_config in enumerate(intra_exp_da_configs):
        exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
        # Get times&performances !!!
        #   Create keys
        key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
        w_node_tile_suffix = f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)'
        key_suffixes = ['_ring', w_node_tile_suffix]
        key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                            for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                                for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
        keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
        #   Parse and select execution times
        raw_time_dict = {key: float(intra_bsa_exe_plans_profile[key]['time']) for key in keys}
        ablation_time_dict = {
            'ring': raw_time_dict[keys[0]], # [TODO] @yqg
            'w_node_tile': raw_time_dict[f'{key_preffix}{w_node_tile_suffix}'],
            # 'w_gpu_tile': raw_time_dict[keys[1]],
            # 'w_kernel_tile': min(raw_time_dict[keys[1]], raw_time_dict[keys[2]]),
            # 'ultra': min([raw_time_dict[key] for key in keys[1:]]),
            'w_gpu_tile': raw_time_dict[keys[-4]],
            'w_gpu+kernel_tile': min(raw_time_dict[keys[-4]], raw_time_dict[keys[-3]]),
            'ultra': min([raw_time_dict[key] for key in keys[-4:]]),
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
    fig.savefig(f"./plot/figs/intra_bsa_configs_inference.pdf", bbox_inches='tight')
  
def main():
    os.environ['CLUSTER_NAME'] = 'zhipu_hamming'
    os.environ['PLATFORM'] = 'H100'
    # prof_db = initialize_prof_db()
    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_infer_configs()
    exp_configs = get_exp_infer_configs()
    
    # Calc all inter_exp_da_configs
    # inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    intra_exp_da_configs = calc_all_intra_exp_da_configs(intra_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    plot_all_intra_configs(intra_exp_da_configs, None)
    
if __name__ == '__main__':
    main()