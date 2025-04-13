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
import regex as re
import random
from plot.common import parse_dense_performance_data
from matplotlib.lines import Line2D

def plot_strong_scalability_for_inference(raw_time_dict: dict):
    fob = 0
    CPs = [
      (2, 1),
      (4, 1),
      (8, 1),
    ]
    # Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
    Ss = [16384, 32768, 65536, 131072]  # all
    Ss = [65536, 131072]  # all
    Ss_str_dict = {
      '16384': '16k',
      '32768': '32k',
      '65536': '64k',
      '131072': '128k',
      '262144': '256k',
      '524288': '512k',
    }
    Nhs = [1, 32]
    Nhs = [32]
    bs = 1
    D = 128
    BSA_reprs = [
      '[[2000][1200][1020][1002]]',
      '[[20000000][12000000][11200000][11120000][10112000][10011200][10001120][10000112]]',
    ]
    BSA_NAMES = [
      'star',
      'streaming'
    ]
    # full_sys_names = ['ring', 'stripe', 'zigzag', 'ultra']  # No w_node_tile yet !!!
    sys_names = ['ring', 'UltraAttn']
    FONT_SIZE = 40
    MARKER_SIZE = 20
    LINEWIDTH = 5
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (20,3.5),  # Column, Row
        'font.sans-serif': 'Times New Roman',
        'axes.labelsize': FONT_SIZE,
        'font.size': 8,
        'legend.fontsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    plt.rcParams.update(figsize)
    num_figs_per_row = len(Nhs) * len(BSA_reprs) * len(Ss) # For cherry pick
    num_figs = num_figs_per_row
    # num_figs_per_row = len(CPs) * len(Nhs)              # For all
    # num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)
    num_rows = num_figs // num_figs_per_row
    fig, axs = plt.subplots(num_rows, num_figs_per_row)
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    marker_def = MARKER_DEF[:len(sys_names)]
    # hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]
    # hatch_def = [None] * len(sys_names)
    
    # 用ABCDEF替代7个sys_name
    # abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # abc = abc[: len(sys_names)]
    bar_width = 0.8
    bar_gap = 0.2
    # bar_width = 0.4
    # bar_gap = 0.1
    # ylim = 1000 # 限制y轴范围, 保持表格整齐
    ylim = 1.1  # Upper bound of relative performance
    
    # ultra_key_suffixes = [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
    #                     for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
    #                         for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    
    # Add ultra suffix for all cases
    cached_keys = list(raw_time_dict.keys())
    for key in cached_keys:
        matched = re.match(r'^(.*)_ring$', key)
        if matched:
            key_prefix = matched.group(1)
            ultra_key = f'{key_prefix}_UltraAttn'
            assert ultra_key not in raw_time_dict.keys()
            # def parse_time_from_suffix(key_suffix):
            #     return float(raw_time_dict[f'{key_prefix}{key_suffix}']['time'])
            def parse_time_from_key(key):
                return float(raw_time_dict[key]['time'])
            all_keys = [key for key in cached_keys if key_prefix in key]
            best_key = min(all_keys, key=parse_time_from_key)
            raw_time_dict[ultra_key] = raw_time_dict[best_key]
    # End
    
    fig_rid = 0
    for S_id, S in enumerate(Ss):
        # fig_rid = S_id
        for bsa_id, bsa_repr in enumerate(BSA_reprs):
            for Nh_id, Nh in enumerate(Nhs):
                fig_cid = (S_id * len(BSA_reprs) + bsa_id) * len(Nhs) + Nh_id
                if num_rows > 1:
                    ax = axs[fig_rid, fig_cid]  # Get subfig
                else:
                    ax = axs[fig_cid]
                # ax.set_aspect(1.0)
                ax.set_box_aspect(1/1.8)  # 宽高比为1.5
                # sub_fig_title = f"Nh={Nh}\n{'fwd' if fob == 0 else 'bwd'}"
                # sub_fig_title = f"Nh={Nh}\n{BSA_NAMES[bsa_id]}"
                sub_fig_title = f"{BSA_NAMES[bsa_id]}\nS={Ss_str_dict[str(S)]}"
                shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                key_preffix_CPs = {}
                for CP in CPs:
                    bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                    key_preffix_CPs[CP] = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                
                time_dict = {}  # {sys_name : [t, t, t, t]}
                for sys_id, sys_name in enumerate(sys_names):
                    if sys_name is None:
                        continue
                    time_dict[sys_name] = [float(raw_time_dict[f'{key_preffix_CPs[CP]}_{sys_name}']['time']) for CP in CPs]
                total_gpu_time = {k: [t * math.prod(CPs[CP_id]) for CP_id, t in enumerate(t_l)] for k, t_l in time_dict.items()}
                total_gpu_time_min = min([t for t_l in total_gpu_time.values() for t in t_l])
                norm_perf = {k: [total_gpu_time_min / t for t in t_l] for k, t_l in total_gpu_time.items()}
                # print(f'norm_perf: {norm_perf}')
                
                x = np.arange(len(CPs)) # 4 for training: (8, 16, 32, 64)
                for sys_id, sys_name in enumerate(sys_names):
                    if sys_name is None:
                        continue
                    ax.plot(x, norm_perf[sys_name], color=pair_color_def[sys_id], marker=marker_def[sys_id], markersize=MARKER_SIZE, linewidth=LINEWIDTH)
                
                ax.set_ylim(0, ylim)
                if fig_cid == 0:
                    ax.set_yticks(np.arange(0, ylim * 2 + 1e-5, 1) / 2)  # [0, 0.5, 1]
                else:
                    ax.set_yticks([])
                
                if fig_rid == num_rows - 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels([math.prod(CP) for CP in CPs])
                else:
                    ax.set_xticks([])
                
                if fig_rid == num_rows - 1:
                    ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-0.88)
                    
                if fig_cid == 0:
                    #   ax.set_ylabel(f'{BSA_NAMES[bsa_id]}\nS={Ss_str_dict[str(S)]}', fontdict={'weight': 'bold'})
                        ax.set_ylabel(f' \n ', fontdict={'weight': 'bold'})
                        ax.yaxis.set_label_coords(-0.4, 0.5)
                      # ax.yaxis.set_label_coords(0, 1)
                for spine in ax.spines.values():
                    spine.set_linewidth(4)  # 设置边框线宽
                # if fig_cid == 0:
                #     # ax.set_ylabel(f'{BSA_NAMES[bsa_id]}\nS={Ss_str_dict[str(S)]}', fontdict={'weight': 'bold'})
                #     ax.set_ylabel(f'S={Ss_str_dict[str(S)]}', fontdict={'weight': 'bold'})
                #     ax.yaxis.set_label_coords(-0.2, 0.5)
                    
    # Add legend to the global fig 
    # legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    legend_handles = [mpatches.Patch(facecolor=pair_color_def[i], edgecolor='k', label=sys_names[i]) for i in range(len(sys_names))]
    # fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.3))
    legend_elements = [
        Line2D([0], [0], marker=marker_def[i], color=pair_color_def[i], linestyle='-', label=sys_names[i], linewidth=LINEWIDTH, markersize=MARKER_SIZE) for i in range(len(sys_names))
    ]
    # fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    fig.savefig(f"./plot/figs/strong_scalability_inference_cherry_pick.pdf", bbox_inches='tight')

                    
def main():
    # BSA for inference time dict
    raw_time_dict = {}
    with open('./database_bsa_infer/zhipu_hamming/H100/intra_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    
    plot_strong_scalability_for_inference(raw_time_dict)
    
    
if __name__ == '__main__':
    main()