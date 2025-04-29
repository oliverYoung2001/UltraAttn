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

def parse_results(file_names, raw_time_dict: dict): # Json like dict
    baselines = ['ring_flash_attn_func', 'zigzag_ring_flash_attn_func', 'stripe_flash_attn_func']
    # skip_lines = 0
    pat0 = re.compile(r'^.*fob=(\d).*$')
    # pat1 = re.compile(r'^SP=\((\d+),(\d+)\),S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat1 = re.compile(r'^.*SP=\((\d+),(\d+)\),.*S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    # mfu: 0.827 Tflops/s, hfu: 0.827 Tflops/s, 6161.278 iter/s, 1.623e-04 s/iter, (4.399, 0.002, 0.001) sec
    pat2 = re.compile(r'^.*hfu: (-?\d*\.?\d+) Tflops/s,.*iter/s, (-?(\d+(?:\.\d+)?(?:e[+-]\d+)?)) s/iter,.*$')
    causal_key_suffixes = [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                                for KERNEL_SCHEDULE_TYPE in ['ILP', 'Flexflow'] \
                                    for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    
    for file_name in file_names:
        with open(file_name, 'r') as f:
            for line in f.readlines():
                # if skip_lines > 0:
                #     skip_lines -= 1
                #     continue
                for baseline in baselines:  # [NOTE]: Results of baseline models are error !!! But why error ???
                    if baseline in line:
                        cur_type = baseline.split('_')[0]	# 'ring', 'zigzag', or 'stripe'
                        continue
                if 'orchestrated_attn_func' in line:
                    if 'fused' in line:
                        continue
                    cur_type = 'ultra'
                    ablation_id = 0
                    continue
                
                res = pat2.match(line)
                if res:
                    # print(f'res: {res.group(0)}, {res.group(1)}')
                    assert key_preffix is not None and fob is not None
                    if cur_type == 'ultra':
                        if causal:
                            key_suffix = causal_key_suffixes[ablation_id]
                        else:
                            YX_num = int(math.log2(CP[1] if CP[1] > 1 else CP[0])) + 1
                            YX_id = ablation_id % YX_num
                            YX = (1 << YX_id, 1 << (YX_num - 1 - YX_id))
                            KERNEL_TILE_TYPE = 'w_kernel_tile' if ablation_id >= YX_num else 'w/o_kernel_tile'
                            key_suffix = f'_ablation=(YX={YX},{KERNEL_TILE_TYPE})'
                        ablation_id += 1
                    else:	# baseline
                        key_suffix = f'_{cur_type}'
                    raw_time_dict[f'{key_preffix}{key_suffix}'] = {
                        'hfu': float(res.group(1)),
                        'time': res.group(2),
                    }
                res = pat1.match(line)
                if res:
                    # print(f'res: {res.group(0)}, {res.group(1)}, {res.group(2)}, {res.group(3)}, {res.group(4)}, {res.group(5)}, {res.group(6)}, {res.group(7)}, {res.group(8)}, {res.group(9)}')
                    CP = (int(res.group(2)), int(res.group(1)))	# (intra, inter)
                    S = (int(res.group(3)), int(res.group(4)))  # just for analysis !!!
                    Nh = (int(res.group(5)), int(res.group(6)))
                    bs = int(res.group(7))
                    D = int(res.group(8))
                    causal = res.group(9) == 'True'
                    bsa_repr = f'[[{2 if causal else 1}]]'
                    shape_config_str = f"S={S}_Nh={Nh}_bs={bs}_D={D}"
                    bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                    key_preffix = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                    continue
                res = pat0.match(line)
                if res:
                    # print(f'res: {res.group(0)}, {res.group(1)}')
                    fob = int(res.group(1))
                    continue


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
    Ss = [16384, 32768, 65536, 131072, 262144, 524288]  # all
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
      'full',
      'causal'
    ]
    # sys_names = ['ring', 'zigzag', 'w_node_tile', 'w_gpu_tile', 'w_kernel_tile', 'ultra']
    sys_names = ['ring', 'stripe', 'zigzag', 'w_node&gpu_tile', 'w_kernel_tile', 'ultra']  # No w_node_tile yet !!!
    figsize = {
        # "figure.figsize": (12,2),  # Column, Row
        # "figure.figsize": (28,3),  # Column, Row
        "figure.figsize": (20,3),  # Column, Row
        "figure.figsize": (20,30),  # Column, Row
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
    # num_figs_per_row = len(CPs) * len(Ss) * len(Nhs)  # For cherry pick
    # num_figs = num_figs_per_row * len(BSA_reprs)
    num_figs_per_row = len(CPs) * len(Nhs)              # For all
    num_figs = num_figs_per_row * len(Ss) * len(BSA_reprs)

    fig, axs = plt.subplots(num_figs // num_figs_per_row, num_figs_per_row)
    # 调整子图之间的横向间隔
    # plt.subplots_adjust(wspace=0)  # 设置横向间隔为 0.1（默认值通常为 0.2）
    
    # 和utils.py中的COLOR_DEF相同，共7种颜色
    pair_color_def = COLOR_DEF[:len(sys_names)]
    hatch_def = [HATCH_DEF[2] if i == len(sys_names) - 1 else None for i in range(len(sys_names))]
    hatch_def = [None] * len(sys_names)

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
                    fig_rid, fig_cid = bsa_id * len(Ss) + S_id, Nh_id * len(CPs) + CP_id
                    
                    shape_config_str = f"S={(S,S)}_Nh={(Nh,Nh)}_bs={bs}_D={D}"
                    bsa_config_str = f'CP={CP}_repr={bsa_repr}'
                    key_preffix = f'fob={fob}_CP={CP}_shape_config={{{shape_config_str}}}_bsa_config={{{bsa_config_str}}}'
                    sub_fig_title = f'CP={CP}\nS={Ss_str_dict[str(S)]},Nh={Nh}'
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
                        ablation_time_dict = {
                            'ring': raw_time_dict[full_keys[0]], # 
                            'stripe': raw_time_dict[full_keys[0]],
                            'zigzag': raw_time_dict[full_keys[0]],  # 
                            # 'w_node_tile': ???,
                            'w_node&gpu_tile': min([raw_time_dict[key] for key in full_keys[1:1+YX_num]]),
                            'w_kernel_tile': min([raw_time_dict[key] for key in full_keys[1:]]),
                            'ultra': min([raw_time_dict[key] for key in full_keys[1:]]),
                        }
                    elif bsa_repr == '[[2]]':   # causal
                        ablation_time_dict = {
                            'ring': raw_time_dict[causal_keys[0]], # 
                            'stripe': raw_time_dict[causal_keys[1]],
                            'zigzag': raw_time_dict[causal_keys[2]],  # 
                            # 'w_node_tile': ???,
                            'w_node&gpu_tile': raw_time_dict[causal_keys[-4]],
                            'w_kernel_tile': min(raw_time_dict[causal_keys[-4]], raw_time_dict[causal_keys[-3]]),
                            'ultra': min([raw_time_dict[key] for key in causal_keys[-4:]]),
                        }
                    else:
                        raise Exception(f'[ERROR]: Unknown bsa_repr={bsa_repr}')
                    
                    ablation_time_list = [ablation_time_dict[sys_name] for sys_name in sys_names]
                    min_time = min(ablation_time_list)
                    norm_perf = [min_time / t for t in ablation_time_list]
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
                    ax.text(bars[-1].get_x() + bars[-1].get_width() / 2 + 0.1, 0.5, f'{norm_perf[-1]/max_baseline:.2f}\u00D7', \
                      fontweight='bold', ha='center', va='center', fontsize=12, color='black', rotation=90)

                    # Labels of the subfig
                    # ax.set_xticks(range(len(abc)), abc)
                    ax.set_xticks([])
                    ax.set_title(sub_fig_title, loc='center', fontsize=10)
                    # if fig_rid == 0:
                    #   ax.set_title(MODEL_NAME[model_name], loc='center', fontsize=10)

                    if fig_cid == 0:
                      ax.set_ylabel(BSA_NAMES[bsa_id], fontsize=12)
                      ax.yaxis.set_label_coords(-0.5, 0.5)
                      # ax.yaxis.set_label_coords(0, 1)

                    ax.set_ylim(0, ylim)
                    if fig_cid == 0:
                        ax.set_yticks(np.arange(0, ylim * 4 + 1, 1) / 4)  # [0, 0.25, 0.5, 0.75, 1]
                    else:
                        ax.set_yticks([])
     
    # Add legend to the global fig 
    # legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label='(' + abc[i] + ') ' + sys_names[i]) for i in range(len(sys_names))]
    legend_handles = [mpatches.Patch(hatch=hatch_def[i], facecolor=pair_color_def[i], edgecolor='k', label=sys_names[i]) for i in range(len(sys_names))]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(sys_names), bbox_to_anchor=(0.5, 1.15))
    # fig.text(0.085, 0.5, 'Relative Performance', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0.5,wspace=0.05)
    fig.savefig(f"./plot/figs/inter_dense_configs_training_fob={fob}.pdf", bbox_inches='tight')

def parse_performance_data():
    INTER_DENSE_EXE_PLANS_PROFILE = f'./plot/results_exp/inter_dense_exe_plans_profile.json'
    if os.path.exists(f'./plot/results_exp/inter_dense_exe_plans_profile.json'):
        print(f'Bypassed')
        with open(INTER_DENSE_EXE_PLANS_PROFILE, 'r') as f:
            raw_time_dict = json.load(f)
    else:
        print(f'Not bypass')
    
        raw_time_dict = {}
        # Data files
        # (8, 1); FULL
        # ./prof_data/fit/wrapper_intra_SP=8_all_final.log
        # (8, 1); CAUSAL
        # ./prof_data/fit/wrapper_intra_SP=8_all_causal.log
        # (8, 2), (8, 4), (8, 8);
        # ./results_exp/fit/A800/final/***
        
        # Parse code for FULL [()_w/o_kernel_tile, ()_w/kernel_tile]

        # Parse code for CAUSAL ['_ablation=(w/o_kernel_tile,ILP)', '_ablation=(w_kernel_tile,ILP)', \
        #                       '_ablation=(w/o_kernel_tile,Flexflow)', '_ablation=(w_kernel_tile,Flexflow)']
        causal_file_names = [
            f'./prof_data/fit/wrapper_intra_SP=8_all_causal.log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=2,8_causal=True_g[15-16].log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=4,8_causal=True_g[13-16].log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=8,8_causal=True_g[07-09,11,13-16].log'
        ]
        full_file_names = [
            f'./prof_data/fit/wrapper_intra_SP=8_all_final.log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=2,8_causal=False_g[07-08].log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=4,8_causal=False_g[13-16].log',
            f'./results_exp/fit/A800/final/wrapper_inter_SP=8,8_causal=False_g[07-09,11,13-16].log'
        ]
        # parse_causal_results(causal_file_names, raw_time_dict)
        # parse_full_results(, raw_time_dict)
        parse_results(causal_file_names + full_file_names, raw_time_dict)
        # [HACK]: calc ring for [(8, 1), full] from [(8, 1), causal] and times (16/15)
        keys = list(raw_time_dict.keys())
        for key in keys:
            # "fob=0_CP=(8, 1)_shape_config={S=(2048, 2048)_Nh=(1, 1)_bs=1_D=128}_bsa_config={CP=(8, 1)_repr=[[2]]}_ring": {
            perf = raw_time_dict[key]
            if str((8, 1)) in key and '[[2]]' in key and 'ring' in key:   # Target causal key
                perf_full = {
                    'hfu': round(perf['hfu'] * 2 / (16 / 15), 3),
                    'time': f"{(float(perf['time']) * (16 / 15)):.3e}",
                }
                key_full = re.sub(r'repr=\[\[2\]\]', 'repr=[[1]]', key)
                assert key_full not in raw_time_dict.keys()
                raw_time_dict[key_full] = perf_full
        # print(f'raw_time_dict: {raw_time_dict}')
        with open(INTER_DENSE_EXE_PLANS_PROFILE, 'w') as f:
            json.dump(raw_time_dict, f)
    raw_times = {k: float(v['time']) for k, v in raw_time_dict.items()}
    return raw_times
  
def main():
    os.environ['CLUSTER_NAME'] = 'hamming'
    os.environ['PLATFORM'] = 'H100'
    # prof_db = initialize_prof_db()
    # inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs()
    # exp_configs = get_exp_configs()
    
    # Calc all inter_exp_da_configs
    # inter_exp_da_configs = calc_all_inter_exp_da_configs(inter_node_bsa_configs, shape_config_dict, exp_configs)
    # End
    raw_time_dict = parse_performance_data()
    plot_all_inter_configs(raw_time_dict, fob=0)
    plot_all_inter_configs(raw_time_dict, fob=1)
    
if __name__ == '__main__':
    main()