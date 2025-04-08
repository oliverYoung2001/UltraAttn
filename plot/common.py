import os
import regex as re
import math
import json

def parse_dense_results(file_names, raw_time_dict: dict): # Json like dict
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

def parse_dense_performance_data():
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
        parse_dense_results(causal_file_names + full_file_names, raw_time_dict)
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
    return raw_times, raw_time_dict
  