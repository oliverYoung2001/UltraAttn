import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from search_algo.initialize import initialize_prof_db
from search_algo.search_engine import Dist_Attn_Config
import json
from typing import List
from search_algo.database import Prof_DB
import regex as re
import random
from plot.common import parse_dense_performance_data
import itertools

def plot_weak_scalability_for_training(raw_time_dict: dict):
    fobs = [0, 1]
    CPs = [
      (8, 1),
      (8, 2),
      (8, 4),
      (8, 8),
    ]
    total_Ss = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
    Ss = [16384, 32768, 65536, 131072, 262144]  # all
    # Ss = [65536, 524288]  # all
    Ss_str_dict = {
      '16384': '16k',
      '32768': '32k',
      '65536': '64k',
      '131072': '128k',
      '262144': '256k',
      '524288': '512k',
      '1048576': '1M',
      '2097152': '2M',
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
        "figure.figsize": (20,90),  # Column, Row
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
    marker_def = MARKER_DEF[:len(full_sys_names)]
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
        for S_id, S_base in enumerate(Ss):
            fig_rid = bsa_id * len(Ss) + S_id
            for fob_id, fob in enumerate(fobs):
                for Nh_id, Nh in enumerate(Nhs):
                    fig_cid = fob_id * len(Nhs) + Nh_id
                    ax = axs[fig_rid, fig_cid]  # Get subfig
                    sub_fig_title = f"Nh={Nh}\n{'fwd' if fob == 0 else 'bwd'}"
                    # calc (S, CP) pairs in one subplot
                    seg_num = 2
                    x_counter = 0
                    x = []
                    xticklabels = []
                    S_CP_pairs = [] # List[List[Tuple]]
                    for CP_base_id, CP_base in enumerate(CPs[: seg_num]):
                        S_CP_pairs.append([])
                        x.append([])
                        log_S_factor = 0
                        while True:
                            CP_id = CP_base_id + log_S_factor * 2
                            if CP_id >= len(CPs):
                                break
                            CP = CPs[CP_id]
                            # CP = CP_base * pow(S_factor, 2)
                            S = S_base * (1 << log_S_factor)
                            # print(f'(S, CP): {(S, CP)}', flush=True)
                            assert S in total_Ss
                            S_CP_pairs[- 1].append((S, CP))
                            x[- 1].append(x_counter)
                            xticklabels.append(math.prod(CP))
                            log_S_factor += 1
                            x_counter += 1
                    # End
                    key_preffix_S_CP_pairs = {}
                    # for S, CP in S_CP_pairs:
                    # for S_CP in itertools.chain.from_iterable(S_CP_pairs):
                    for seg_pairs in S_CP_pairs:
                        for S, CP in seg_pairs:
                            shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                            bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                            key_preffix_S_CP_pairs[(S, CP)] = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                    
                    time_dict = {}  # {sys_name : [t, t, t, t]}
                    norm_perf = {}
                    for sys_id, sys_name in enumerate(sys_names):
                        if sys_name is None:
                            continue
                        # time_dict[sys_name] = [float(raw_time_dict[f'{key_preffix_CPs[CP]}_{sys_name}']['time']) for CP in CPs]
                        time_dict[sys_name] = [[float(raw_time_dict[f'{key_preffix_S_CP_pairs[S_CP]}_{sys_name}']['time']) for S_CP in seg_pairs] \
                                                for seg_pairs in S_CP_pairs]    # List[List[float]]
                        norm_perf[sys_name] = [[] for seg_pairs in S_CP_pairs]
                    # print(f'S_CP_pairs: {S_CP_pairs}', flush=True)
                    for seg_id in range(seg_num):
                        # print(f'seg-id: {seg_id}; sys_names: {sys_names}', flush=True)
                        # print(f'time_dict: {time_dict}', flush=True)
                        total_gpu_time_per_comp_unit_min = min([t for sys_name in sys_names if sys_name is not None for t in time_dict[sys_name][seg_id]])
                        for sys_name in sys_names:
                            if sys_name is not None:
                                norm_perf[sys_name][seg_id] = [total_gpu_time_per_comp_unit_min / t for t in time_dict[sys_name][seg_id]]
                    # print(f'norm_perf: {norm_perf}')
                    # 数据
                    # x = np.linspace(0, 10, 100)
                    # x = np.arange(len(CPs)) # 4 for training: (8, 16, 32, 64)
                    # x = np.arange(len(S_CP_pairs))  # 4 for training: (8, 32, 16, 64)
                    
                    for sys_id, sys_name in enumerate(sys_names):
                        if sys_name is None:
                            continue
                        for seg_id, seg_norm_perf in enumerate(norm_perf[sys_name]):
                            ax.plot(x[seg_id], seg_norm_perf, color=pair_color_def[sys_id], marker=marker_def[sys_id], markersize=20, linewidth=6)
                    
                    ax.set_ylim(0, ylim)
                    if fig_cid == 0:
                        ax.set_yticks(np.arange(0, ylim * 2 + 1e-5, 1) / 2)  # [0, 0.5, 1]
                    else:
                        ax.set_yticks([])
                    
                    if fig_rid == num_rows - 1:
                        ax.set_xticks(np.arange(len(CPs)))
                        # ax.set_xticklabels([math.prod(CP) for CP in CPs])   # [TODO]
                        ax.set_xticklabels(xticklabels)
                    else:
                        ax.set_xticks([])
                    
                    if fig_rid == num_rows - 1:
                        ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-1)

                    if fig_cid == 0:
                      ax.set_ylabel(f'{BSA_NAMES[bsa_id]}\nS_base={Ss_str_dict[str(S_base)]}', fontdict={'weight': 'bold'})
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
    fig.savefig(f"./plot/figs/weak_scalability_training.pdf", bbox_inches='tight')

                    
def main():
    _, raw_time_dict = parse_dense_performance_data()  # {key: {'hfu': xxx, 'time': xxx, 'sim_time': xxx}}}
    # BSA for training time dict
    with open('./database_bsa_train/hamming/H100/inter_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    
    plot_weak_scalability_for_training(raw_time_dict)
    
    
if __name__ == '__main__':
    main()