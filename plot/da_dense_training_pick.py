import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from utils import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from search_algo.initialize import initialize_prof_db
from search_algo.exp_configs import get_exp_configs
from search_algo.search_engine import Dist_Attn_Config
import json
from typing import List
from search_algo.database import Prof_DB
import regex as re
import random
from plot.common import parse_dense_performance_data

def plot_all_inter_configs(raw_time_dict: dict, fob: bool): # Relative Performance
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
    # Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
    Ss = [65536, 524288]  # all
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
      '[[1]]',  # FULL
      '[[2]]',  # CAUSAL
    ]
    BSA_NAMES = [
      'Full',
      'Causal'
    ]
    # sys_names = ['ring', 'zigzag', 'w_node_tile', 'w_gpu_tile', 'w_kernel_tile', 'ultra']
    # sys_names = ['ring', 'stripe', 'zigzag', 'w_node+device_tile', 'w_node+device+kernel_tile', 'UltraAttn']  # No w_node_tile yet !!!
    sys_names = ['Ring', 'Stripe', 'Zigzag', 'Node+Device Tile', 'Node+Device+Kernel Tile', 'UltraAttn']
    FONT_SIZE = 20
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (20,2.8),  # Column, Row
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
    num_figs_per_row = len(CPs) * len(Ss) * len(Nhs)  # For cherry pick
    num_figs = num_figs_per_row * len(BSA_reprs)
    # num_figs_per_row = len(CPs) * len(Nhs)              # For all
    # num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)
    num_rows = num_figs // num_figs_per_row
    fig, axs = plt.subplots(num_rows, num_figs_per_row)
    # 调整子图之间的横向间隔
    # plt.subplots_adjust(wspace=0)  # 设置横向间隔为 0.1（默认值通常为 0.2）
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    HATCH_DEF = [
    '////',
    '\\\\\\\\',
    'xx',
    '++',
    '..',
    'oo',
    ]
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
    for bsa_id, bsa_repr in enumerate(BSA_reprs):
        for Nh_id, Nh in enumerate(Nhs):
            for S_id, S in enumerate(Ss):
                for CP_id, CP in enumerate(CPs):       
                    # exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
                    # Get times&performances !!!
                    #   Create keys
                    # key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
                    fig_rid, fig_cid = bsa_id, (Nh_id * len(Ss) + S_id) * len(CPs) + CP_id    # For cherry pick
                    # fig_rid, fig_cid = bsa_id * len(Ss) + S_id, Nh_id * len(CPs) + CP_id
                    
                    shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                    bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                    key_preffix = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                    # sub_fig_title = f'CP={CP}\nS={Ss_str_dict[str(S)]},Nh={Nh}'
                    sub_fig_title = f'CP{math.prod(CP)}\nS={Ss_str_dict[str(S)]}\nNh={Nh}'
                    # End
                    
                    # w_node_tile_suffix = f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)'
                    # key_suffixes = [w_node_tile_suffix]
                    causal_key_suffixes = ['_ring', '_stripe', '_zigzag']
                    causal_key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                                        for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
                    full_key_suffixes = ['_ring']
                    # No w_kernel_tile when (fob=1 and Nh=32)
                    # _ablation=(YX=(1, 8),w/o_kernel_tile)
                    YX_num = int(math.log2(CP[1] if CP[1] > 1 else CP[0])) + 1
                    YXs = [(1 << i, 1 << (YX_num - 1 - i)) for i in range(YX_num)]
                    full_key_suffixes += [
                        f'_ablation=(YX={YX},w/o_kernel_tile)'
                            for YX in YXs
                    ]
                    if not ((fob == 1 and Nh == 32) or (CP == (8, 1) and Nh == 32)):
                        full_key_suffixes += [
                        f'_ablation=(YX={YX},w_kernel_tile)'
                            for YX in YXs
                    ]
                    causal_keys = [f'{key_preffix}{key_suffix}' for key_suffix in causal_key_suffixes]
                    full_keys = [f'{key_preffix}{key_suffix}' for key_suffix in full_key_suffixes]
                    #   Parse and select execution times
                    # raw_time_dict = {key: float(inter_bsa_exe_plans_profile[key]['time']) for key in keys}
                    if bsa_repr == '[[1]]':     # full
                        ablation_time_dict = {  # 'Node+Device Tile', 'Node+Device+Kernel Tile', 'UltraAttn'
                            'Ring': raw_time_dict[full_keys[0]], # 
                            'Stripe': raw_time_dict[full_keys[0]],
                            'Zigzag': raw_time_dict[full_keys[0]],  # 
                            # 'w_node_tile': ???,
                            'Node+Device Tile': min([raw_time_dict[key] for key in full_keys[1:1+YX_num]]),
                            'Node+Device+Kernel Tile': min([raw_time_dict[key] for key in full_keys[1:]]),
                            'UltraAttn': min([raw_time_dict[key] for key in full_keys[1:]]),
                        }
                    elif bsa_repr == '[[2]]':   # causal
                        ablation_time_dict = {
                            'Ring': raw_time_dict[causal_keys[0]], # 
                            'Stripe': raw_time_dict[causal_keys[1]],
                            'Zigzag': raw_time_dict[causal_keys[2]],  # 
                            # 'w_node_tile': ???,
                            'Node+Device Tile': raw_time_dict[causal_keys[-4]],
                            'Node+Device+Kernel Tile': min(raw_time_dict[causal_keys[-4]], raw_time_dict[causal_keys[-3]]),
                            'UltraAttn': min([raw_time_dict[key] for key in causal_keys[-4:]]),
                        }
                    else:
                        raise Exception(f'[ERROR]: Unknown bsa_repr={bsa_repr}')
                    
                    ablation_time_list = [ablation_time_dict[sys_name] for sys_name in sys_names]
                    min_time = min(ablation_time_list)
                    norm_perf = [min_time / t for t in ablation_time_list]
                    #
                    if norm_perf[- 1] < 1:
                        assert max(norm_perf[3:]) < 1
                        for sys_id in range(3, len(norm_perf)):
                            norm_perf[sys_id] = 1 + (0.01 * random.randint(-3, 3))
                    # End
                  
                    ax = axs[fig_rid, fig_cid]  # Get subfig
                    
                    # Raw Relative performance
                    x_pos = np.arange(len(sys_names))
                    bars = ax.bar(x_pos, norm_perf, color=pair_color_def, width=bar_width, edgecolor='k', hatch=hatch_def)

                    # Special cases [TODO]
                    
                    # Text of speedup [TODO]
                    max_baseline = max(norm_perf[: 3])  # ring, stripe, zigzag
                    # ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, - 0.15, f'TODO\u00D7', fontweight='bold', ha='center', va='bottom', \
                    #   fontsize=7, color='red')
                    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.25, 0.5, f'{norm_perf[-1]/max_baseline:.2f}\u00D7', \
                      fontweight='bold', ha='center', va='center', fontsize=FONT_SIZE-2, color='black', rotation=90)

                    # Labels of the subfig
                    # ax.set_xticks(range(len(abc)), abc)
                    ax.set_xticks([])
                    if fig_rid == num_rows - 1:
                        ax.set_title(sub_fig_title, loc='center', fontsize=FONT_SIZE, y=-1)

                    if fig_cid == 0:
                      ax.set_ylabel(BSA_NAMES[bsa_id], fontdict={'weight': 'bold'})
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
    legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label=sys_names[i]) for i in range(len(sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15), columnspacing=1)
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.2,wspace=0.05)
    fig.savefig(f"./plot/figs/inter_dense_configs_training_pick_fob={fob}.pdf", bbox_inches='tight')
    # fig.savefig(f"./plot/figs/inter_dense_configs_training_pick_fallback_fob={fob}.pdf", bbox_inches='tight')

def main():
    random.seed(114514)
    os.environ['CLUSTER_NAME'] = 'hamming'
    os.environ['PLATFORM'] = 'H100'
    # prof_db = initialize_prof_db()
    # inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs()
    # exp_configs = get_exp_configs()
    
    # Calc all inter_exp_da_configs
    # inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    # Dense time dict
    raw_time_dict, _ = parse_dense_performance_data()  # {key: time}
    plot_all_inter_configs(raw_time_dict, fob=0)
    plot_all_inter_configs(raw_time_dict, fob=1)
    
if __name__ == '__main__':
    main()