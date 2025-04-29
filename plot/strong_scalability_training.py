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
    
def plot_strong_scalability_for_training(raw_time_dict: dict):
    fobs = [0, 1]
    CPs = [
      (8, 1),
      (8, 2),
      (8, 4),
      (8, 8),
    ]
    Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
    # Ss = [65536, 524288]  # all
    Ss_str_dict = {
      '16384': '16k',
      '32768': '32k',
      '65536': '64k',
      '131072': '128k',
      '262144': '256k',
      '524288': '512k',
    }
    Nhs = [1, 32]
    bs = 1
    D = 128
    BSA_reprs = [
        '[[1110000000000000][1111000000000000][1111000000000000][0111000000000000][0000111000000000][0000111100000000][0000111100000000][0000011100000000][0000000011100000][0000000011110000][0000000011110000][0000000001110000][0000000000001110][0000000000001111][0000000000001111][0000000000000111]]',
        '[[1111111111111111][1100000000000000][1010000000000000][1001000000000000][1000100000000000][1000010000000000][1000001000000000][1000000100000000][1000000010000000][1000000001000000][1000000000100000][1000000000010000][1000000000001000][1000000000000100][1000000000000010][1000000000000001]]',
        '[[1]]',  # FULL
        '[[2]]',  # CAUSAL
    ]
    BSA_NAMES = [
        'strided',
        'global+local',
        'full',
        'causal',
    ]
    full_sys_names = ['ring', 'stripe', 'zigzag', 'ultra']  # No w_node_tile yet !!!
    sub_sys_names = ['ring', None, None, 'ultra']
    FONT_SIZE = 20
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (20,100),  # Column, Row
        'font.sans-serif': 'Times New Roman',
        'axes.labelsize': FONT_SIZE,
        'font.size':8,
        'legend.fontsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': 15,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    plt.rcParams.update(figsize)
    num_figs_per_row = len(Nhs) * len(fobs) # For cherry pick
    num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)
    # num_figs_per_row = len(CPs) * len(Nhs)              # For all
    # num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)
    num_rows = num_figs // num_figs_per_row
    fig, axs = plt.subplots(num_rows, num_figs_per_row)
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(full_sys_names)]
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
    ylim = 1  # Upper bound of relative performance
    
    # ultra_key_suffixes = [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
    #                     for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
    #                         for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    
    # Add ultra suffix for all cases
    cached_keys = list(raw_time_dict.keys())
    for key in cached_keys:
        matched = re.match(r'^(.*)_ring$', key)
        if matched:
            key_prefix = matched.group(1)
            ultra_key = f'{key_prefix}_ultra'
            assert ultra_key not in raw_time_dict.keys()
            # def parse_time_from_suffix(key_suffix):
            #     return float(raw_time_dict[f'{key_prefix}{key_suffix}']['time'])
            def parse_time_from_key(key):
                return float(raw_time_dict[key]['time'])
            all_keys = [key for key in cached_keys if key_prefix in key]
            best_key = min(all_keys, key=parse_time_from_key)
            raw_time_dict[ultra_key] = raw_time_dict[best_key]
    # End
    
    for bsa_id, bsa_repr in enumerate(BSA_reprs):
        if bsa_repr in ['[[2]]']:           # Causal
            sys_names = full_sys_names
        else:                               # Others
            sys_names = sub_sys_names
        for S_id, S in enumerate(Ss):
            fig_rid = bsa_id * len(Ss) + S_id
            for fob_id, fob in enumerate(fobs):
                for Nh_id, Nh in enumerate(Nhs):
                    fig_cid = fob_id * len(Nhs) + Nh_id
                    ax = axs[fig_rid, fig_cid]  # Get subfig
                    sub_fig_title = f"Nh={Nh}\n{'fwd' if fob == 0 else 'bwd'}"
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
                    # 数据
                    # x = np.linspace(0, 10, 100)
                    x = np.arange(len(CPs)) # 4 for training: (8, 16, 32, 64)
                    for sys_id, sys_name in enumerate(sys_names):
                        if sys_name is None:
                            continue
                        ax.plot(x, norm_perf[sys_name], color=pair_color_def[sys_id], linewidth=6)
                    # y1 = np.sin(x)
                    # y2 = np.cos(x)
                    # y3 = np.tan(x) / 10  # 缩小 tan 的幅度，避免过大

                    # # 绘制折线并设置标签
                    # ax.plot(x, y1, label='Sine', color='blue', linewidth=2)
                    # ax.plot(x, y2, label='Cosine', color='red', linestyle='--', linewidth=2)
                    # ax.plot(x, y3, label='Tangent', color='green', linestyle='-.', linewidth=2)

                    # 添加图例
                    # ax.legend()

                    # 添加标题和轴标签
                    # ax.set_title('Multiple Lines with Legend')
                    # ax.set_xlabel('X Axis')
                    # ax.set_ylabel('Y Axis')
                    
                    ax.set_ylim(0, ylim)
                    if fig_cid == 0:
                        ax.set_yticks(np.arange(0, ylim * 2 + 1, 1) / 2)  # [0, 0.5, 1]
                    else:
                        ax.set_yticks([])
                    
                    if fig_rid == num_rows - 1:
                        ax.set_xticks(x)
                        ax.set_xticklabels([math.prod(CP) for CP in CPs])
                    else:
                        ax.set_xticks([])
                    
                    if fig_rid == num_rows - 1:
                        ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-1)

                    if fig_cid == 0:
                      ax.set_ylabel(f'{BSA_NAMES[bsa_id]}\nS={Ss_str_dict[str(S)]}', fontdict={'weight': 'bold'})
                      ax.yaxis.set_label_coords(-0.5, 0.5)
                      # ax.yaxis.set_label_coords(0, 1)

                    # 显示网格（可选）
                    # plt.grid(True)

                    # plt.show()
                    
    # Add legend to the global fig 
    # legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    legend_handles = [mpatches.Patch(facecolor=pair_color_def[i], edgecolor='k', label=sys_names[i]) for i in range(len(sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.05)
    fig.savefig(f"./plot/figs/strong_scalability_training.pdf", bbox_inches='tight')

                    
def main():
    _, raw_time_dict = parse_dense_performance_data()  # {key: {'hfu': xxx, 'time': xxx, 'sim_time': xxx}}}
    # BSA for training time dict
    with open('./database_bsa_train/hamming/H100/inter_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    
    plot_strong_scalability_for_training(raw_time_dict)
    
    
if __name__ == '__main__':
    main()