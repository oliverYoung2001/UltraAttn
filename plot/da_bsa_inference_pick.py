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

def plot_all_intra_configs(intra_exp_da_configs, prof_db: Prof_DB): # Relative Performance
    """
    intra_exp_da_configs: List[{'exp_config': xxx, 'da_config': xxx}]
    """
    fob = 0
    CPs = [
      (2, 1),
      (4, 1),
      (8, 1),
    ]
    # Ss = [16384, 65536]  # 16K, 64K
    Ss = [16384, 32768]  # 16K, 32K
    Ss = [65536, 131072]  # 16K, 32K
    # Ss = [262144]  # 16K, 32K
    Ss = [32768, 131072]  # 16K, 32K
    Ss_str_dict = {
      '16384': '16k',
      '32768': '32k',
      '65536': '64k',
      '131072': '128k',
      '262144': '256k',
      '524288': '512k',
    }
    Nhs = [32]
    bs = 1
    D = 128
    BSA_reprs = [
      '[[2000][1200][1020][1002]]',
      '[[20000000][12000000][11200000][11120000][10112000][10011200][10001120][10000112]]',
    ]
    BSA_NAMES = [
      'Star',
      'Streaming'
    ]
    # with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'r') as f:
    with open('./database_bsa_infer/hamming/H100/intra_bsa_exe_plans_profile.json', 'r') as f:
      intra_bsa_exe_plans_profile = json.load(f)
    FONT_SIZE = 22
    # sys_names = ['ring', 'w_device_tile', 'w_device+kernel_tile', 'UltraAttn']
    sys_names = ['Ring', 'Device Tile', 'Device+Kernel Tile', 'UltraAttn']
    figsize = {
        "figure.figsize": (12,3),
        "figure.figsize": (10,2.8),
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
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    hatch_def = [None] * len(sys_names)
    hatch_def = HATCH_DEF[:len(sys_names)-1] + [None]

    # 用ABCDEF替代7个sys_name
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    abc = abc[: len(sys_names)]
    bar_width = 0.8
    bar_gap = 0.2
    # ylim = 1000 # 限制y轴范围, 保持表格整齐
    ylim = 1  # Upper bound of relative performance

    # for c_id, exp_da_config in enumerate(intra_exp_da_configs):
    for fig_rid, bsa_repr in enumerate(BSA_reprs):
        fig_cid = - 1
        for Nh in Nhs:
            for S in Ss:
                for CP in CPs:
                    fig_cid += 1
                    # Get times&performances !!!
                    #   Create keys
                    # key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
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
                    raw_time_dict = {key: float(intra_bsa_exe_plans_profile[key]['time']) for key in keys}
                    ablation_time_dict = {  # 'Device Tile', 'Device+Kernel Tile'
                        'Ring': raw_time_dict[keys[0]], # @yqg
                        # 'w_node_tile': raw_time_dict[f'{key_preffix}{w_node_tile_suffix}'],
                        # 'w_gpu_tile': raw_time_dict[keys[1]],
                        # 'w_gpu+kernel_tile': min(raw_time_dict[keys[1]], raw_time_dict[keys[2]]),
                        # 'ultra': min([raw_time_dict[key] for key in keys[1:]]),
                        'Device Tile': raw_time_dict[keys[-4]],
                        'Device+Kernel Tile': min(raw_time_dict[keys[-4]], raw_time_dict[keys[-3]]),
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
                    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, 0.5, f'{norm_perf[-1]/max_baseline:.2f}\u00D7', fontweight='bold', ha='center', va='center', \
                      fontsize=FONT_SIZE, color='black', rotation=90)

                    # Labels of the subfig
                    # ax.set_xticks(range(len(abc)), abc)
                    ax.set_xticks([])
                    if fig_rid == num_rows - 1:
                        ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-1)

                    if fig_cid == 0:
                      ax.set_ylabel(BSA_NAMES[fig_rid], fontdict={'weight': 'bold'})
                      ax.yaxis.set_label_coords(-0.35, 0.5)
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
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15), columnspacing=0.2, handletextpad=0.1)
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.05)
    fig.savefig(f"./plot/figs/intra_bsa_configs_inference_pick.pdf", bbox_inches='tight')
  
def main():
    os.environ['CLUSTER_NAME'] = 'hamming'
    os.environ['PLATFORM'] = 'H100'
    # prof_db = initialize_prof_db()
    # inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_infer_configs()
    # exp_configs = get_exp_infer_configs()
    
    # Calc all inter_exp_da_configs
    # inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    # intra_exp_da_configs = calc_all_intra_exp_da_configs(intra_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    plot_all_intra_configs(None, None)
    
if __name__ == '__main__':
    main()