import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from plot.common import parse_dense_performance_data
import json
import regex as re
from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

time_wo_attn_training = {        # Unit: ms
    (1, (8, 1)): [28.9, 30.7, 31.4, 35.9, 53.5, 102.9],    
    (1, (8, 2)): [29.3, 29.6, 28.6, 32.9, 36.4, 55.4, 102.0], 
    (1, (8, 4)): [34.2, 31.4, 34.5, 37.8, 34.0, 38.8, 55.8, 106.7], 
    (1, (8, 8)): [36.7, 39.3, 35.1, 41.4, 37.8, 48.9, 41.8, 61.0], 
    (32, (8, 1)): [47.3, 70.7, 118.0, 218.4, 413.8, 817.8],    
    (32, (8, 2)): [45.2, 48.3, 68.4, 117.1, 215.7, 413.5, 815.6], 
    (32, (8, 4)): [44.7, 42.0, 52.6, 68.8, 116.0, 216.5, 412.2, 818.2], 
    (32, (8, 8)): [48.8, 53.0, 53.0, 54.3, 78.3, 119.6, 217.6, 416.6], 
}

time_wo_attn_inference = {        # Unit: ms
    (1, (2, 1)): [18.4, 18.4, 20.2, 18.5],        # from 16K
    (1, (4, 1)): [18.6, 18.4, 18.7, 22.2, 18.4], 
    (1, (8, 1)): [20.8, 20.9, 19.3, 18.5, 18.6, 21.3], 
    (32, (2, 1)): [28.6, 52.1, 93.6, 194.0],    
    (32, (4, 1)): [20.0, 31.6, 50.1, 93.5, 194.0], 
    (32, (8, 1)): [19.5, 20.8, 30.4, 53.2, 97.0, 204,8], 
}


def plot_e2e(raw_time_dict):
    # fobs = [0, 1]
    CPs_inference = [
      (2, 1),
      (4, 1),
      (8, 1),
    ]
    CPs_training = [
      (8, 1),
      (8, 2),
      (8, 4),
      (8, 8),
    ]
    S_PER_GPU_UB = 2097152
    S_list = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    Ss = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]  # all
    Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
    Ss = [524288]  # all
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
        '[[2000][1200][1020][1002]]',
        '[[20000000][12000000][11200000][11120000][10112000][10011200][10001120][10000112]]',
    ]
    BSA_NAMES = [
        'Strided',
        'Global\n+Local',
        'Full',
        'Causal',
        'Star',
        'Streaming',
    ]
    full_sys_names = ['Ring', 'Stripe', 'Zigzag', 'UltraAttn']  # For causal
    sub_sys_names = ['Ring', 'UltraAttn']   # For others
    
    FONT_SIZE = 25
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (10,2.8),  # Column, Row
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
    num_figs_per_row = len(BSA_reprs) # For cherry pick
    num_figs = num_figs_per_row * len(Ss) * len(Nhs)
    # num_figs_per_row = len(CPs) * len(Nhs)              # For all
    # num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)
    num_rows = num_figs // num_figs_per_row
    # fig, axs = plt.subplots(num_rows, num_figs_per_row)
    fig = plt.figure()
    gs = fig.add_gridspec(num_rows, num_figs_per_row, width_ratios=[1, 1, 1, 2, 1, 1])
    axs = []
    for r_id in range(num_rows):
        axs.append([])
        for c_id in range(num_figs_per_row):
            axs[-1].append(fig.add_subplot(gs[r_id, c_id]))

    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = np.array(COLOR_DEF[:len(full_sys_names)])
    hatch_def = [HATCH_DEF[2] if i == len(full_sys_names) - 1 else None for i in range(len(full_sys_names))]
    hatch_def = np.array([None] * len(full_sys_names))
    hatch_def = np.array(HATCH_DEF[:len(full_sys_names)-1] + [None])

    # 用ABCDEF替代7个sys_name
    # abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # abc = abc[: len(sys_names)]
    bar_width = 0.8
    bar_gap = 0.2
    # bar_width = 0.4
    # bar_gap = 0.1
    # ylim = 1000 # 限制y轴范围, 保持表格整齐
    ylim = 1  # Upper bound of relative performance
    
    # Add ultra suffix for all cases
    cached_keys = list(raw_time_dict.keys())
    for key in cached_keys:
        matched = re.match(r'^(.*)_ring$', key)
        if matched:
            key_prefix = matched.group(1)
            ultra_key = f'{key_prefix}_{full_sys_names[-1].lower()}'
            assert ultra_key not in raw_time_dict.keys()
            def parse_time_from_key(key):
                return float(raw_time_dict[key]['time'])
            all_keys = [key for key in cached_keys if key_prefix in key]
            best_key = min(all_keys, key=parse_time_from_key)
            raw_time_dict[ultra_key] = raw_time_dict[best_key]
    # End
    N_LAYERS = 4
    
    for Nh_id, Nh in enumerate(Nhs):
        for S_id, S in enumerate(Ss):
            fig_rid = Nh_id * len(Ss) + S_id
            for bsa_id, bsa_repr in enumerate(BSA_reprs):
                fig_cid = bsa_id
                if bsa_repr in ['[[2]]']:           # Causal
                    sys_names = full_sys_names
                    sys_ids = np.arange(len(full_sys_names))
                else:                               # Others
                    sys_names = sub_sys_names
                    sys_ids = [full_sys_names.index(sys_name) for sys_name in sys_names]
                if bsa_id < 4:  # Training
                    CP = CPs_training[-1]
                    fobs = [0, 1]
                    time_wo_attention = time_wo_attn_training[(Nh, CP)][S_list.index(S)] / 1000  # Unit: s
                else:   # Inference
                    CP = CPs_inference[-1]
                    fobs = [0]
                    time_wo_attention = time_wo_attn_inference[(Nh, CP)][S_list.index(S)] / 1000  # Unit: s
                if S // math.prod(CP) > S_PER_GPU_UB:
                    continue
                sub_fig_title = f'{BSA_NAMES[bsa_id]}'
                shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                key_preffixes = [f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}' for fob in fobs]
                
                times_attention = np.array([ # Unit: s
                    sum([float(raw_time_dict[f'{key_preffixes[fob]}_{sys_name.lower()}']['time']) for fob in fobs]) * N_LAYERS
                        for sys_name in sys_names
                ])
                e2e_time = times_attention + time_wo_attention
                norm_e2e_prof = min(e2e_time) / e2e_time
                
                ax = axs[fig_rid][fig_cid]  # Get subfig
                    
                # Raw Relative performance
                x_pos = np.arange(len(sys_names))
                bars = ax.bar(x_pos, norm_e2e_prof, color=pair_color_def[sys_ids], width=bar_width, edgecolor='k', hatch=hatch_def[sys_ids])

                # Special cases [TODO]
                
                # Text of speedup [TODO]
                max_baseline = max(norm_e2e_prof[: -1])
                # ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, - 0.15, f'TODO\u00D7', fontweight='bold', ha='center', va='bottom', \
                #   fontsize=7, color='red')
                ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, 0.5, f'{norm_e2e_prof[-1]/max_baseline:.2f}\u00D7', fontweight='bold', ha='center', va='center', \
                    fontsize=FONT_SIZE, color='black', rotation=90)

                # Labels of the subfig
                # ax.set_xticks(range(len(abc)), abc)
                ax.set_xticks([])
                if fig_rid == num_rows - 1:
                    # ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-0.6-(0.2 if '\n' in BSA_NAMES[bsa_id] else 0))
                    ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-0.45-(0.35 if '\n' in BSA_NAMES[bsa_id] else 0))
                # if fig_rid == 0:
                #   ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

                if fig_cid == 0:
                    ax.set_ylabel(f'Nh={Nh}', fontdict={'weight': 'bold'})
                    ax.yaxis.set_label_coords(-0.5, 0.5)
                    # ax.yaxis.set_label_coords(0, 1)

                ax.set_ylim(0, ylim * 1.05)
                if fig_cid == 0:
                    # ax.set_yticks(np.arange(0, ylim * 4 + 1, 1) / 4)  # [0, 0.25, 0.5, 0.75, 1]
                    ax.set_yticks(np.arange(0, ylim * 2 + 1, 1) / 2)  # [0, 0.5, 1]
                else:
                    ax.set_yticks([])
    # Add legend to the global fig 
    # legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label=full_sys_names[i]) for i in range(len(full_sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(full_sys_names), bbox_to_anchor=(0.5, 1.2), columnspacing=0.5)
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.05)
    fig.savefig(f"./plot/figs/e2e_pick.pdf", bbox_inches='tight')
    
def main():
    _, raw_time_dict = parse_dense_performance_data()  # {key: {'hfu': xxx, 'time': xxx, 'sim_time': xxx}}}
    # BSA for training time dict
    with open('./database_bsa_train/hamming/H100/inter_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    # BSA for inference time dict
    with open('./database_bsa_train/hamming/H100/inter_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    with open('./database_bsa_infer/hamming/H100/intra_bsa_exe_plans_profile.json', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    raw_time_dict.update(inter_bsa_exe_plans_profile)
    
    plot_e2e(raw_time_dict)

if __name__ == '__main__':
    main()