import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, get_cc_optimal_schedule_from_table
from search_algo.dependent_graph import Dependent_Graph
from search_algo.graph_transformation_engine import Graph_Transformation_Engine
from search_algo.execute_plan import Execution_Plan, Fused_Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from search_algo.bsa_config import BSA_Repr, BSA_Config
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs
from search_algo.utils import combine_list_to_0, convert_block_table_to_value, parse_args, print_rank_0, report_memory
from search_algo.database import Prof_DB
import torch
import torch.distributed as dist
from functools import partial
import socket
import tests
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from search_algo.workload_partition import solve_sparse_from_bsa
from search_algo.benchmark import benchmark_orchestrate_bsa, prepare_inter_comp_plans
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func
import json
import random
from typing import List, Tuple, Union, Optional
from search_algo.exp_configs import step0_top_down_decompose
from search_algo.initialize import initialize_distribution, initialize_prof_db
from orchestrated_attn.global_vars import set_global_var as set_global_var_orch
from orchestrated_attn.global_vars import get_global_var as get_global_var_orch
from search_algo.dense_utils import create_ablation_configs_for_full
    
# In-file global vars
DTYPE = torch.bfloat16
# End

def profile_all_inter_BSA(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor, prof_db: Prof_DB):
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    print_rank_0(f'key_preffix: {key_preffix}')

    PROC_INFO = get_global_var(f'PROC_INFO')
    # [TODO]: Support baseline here !!! @yqg
    # baseline_funcs = [
    #     ring_flash_attn_func,
    #     zigzag_ring_flash_attn_func,      # baseline
    # ]
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    rank = PROC_INFO['rank']
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'

    # Bypass useless node !!!
    hierarchy_cp = da_config.hierarchy_sp
    if hierarchy_cp > node_num:
        print_rank_0(f'[WARN]: Current task needs {hierarchy_cp} nodes, but now there are only {node_num} nodes !!!')
        return
    if node_id >= hierarchy_cp:
        return
    # Correct gloo_global_group&ncclcomm_global for current task
    sub_group_key = tuple(range(hierarchy_cp * local_size))
    assert rank in sub_group_key
    
    cpu_group_dict = get_global_var_orch('cpu_group_dict')
    if sub_group_key not in cpu_group_dict.keys():
        assert False
        cpu_group_dict[sub_group_key] = torch.distributed.new_group(sub_group_key, backend='gloo')
        set_global_var_orch('cpu_group_dict', cpu_group_dict)
    gloo_global_group = cpu_group_dict[sub_group_key]
    torch.distributed.barrier(group=gloo_global_group)
    
    ncclcomm_dict = get_global_var_orch('ncclcomm_dict')
    if sub_group_key not in ncclcomm_dict.keys():
        assert False
        ncclcomm_dict[sub_group_key] = PyNcclCommunicator(gloo_global_group, ranks=sub_group_key, device=torch.cuda.current_device())
        set_global_var_orch('ncclcomm_dict', ncclcomm_dict)
    ncclcomm_global = ncclcomm_dict[sub_group_key]
    # End
    
    # experiment variables
    WARMUP, NUM_ITER = 11, 20 # most, best performance for most cases
    WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    
    # Prepare database
    with open(prof_db.INTER_BSA_EXE_PLANS_KV, 'r') as f:
        inter_bsa_exe_plans_dict = json.load(f)
    with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    # Create keys
    key_suffixes = [f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)']
    key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                        for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
    # Generate inter_comp_plans_dicts
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    sim_times = []
    for key in keys:
        if key in inter_bsa_exe_plans_profile.keys():   # Bypassed
            continue
        # load exe_plan
        plan_id = inter_bsa_exe_plans_dict[key]
        with open(f'{prof_db.INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
            inter_bsa_execution_plan: Execution_Plan = pickle.load(fin)
        sim_times.append(inter_bsa_execution_plan.end_time)
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        # inter_comp_plans_dict: {intra_bsa_key: intra_bsa_exe_plan}
        # intra_bsa_key = ((relative_Sq, relative_Skv), str(da_config.bsa_config.bsa_repr))
        inter_comp_plans_dict = prepare_inter_comp_plans(inter_bsa_execution_plan, prof_db, da_config)
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
    
    # is_bypass = all([key in inter_bsa_exe_plans_profile.keys() for key in keys])
    is_bypass = len(inter_bsa_execution_plans) == 0
    if not is_bypass:
        print_rank_0(f'Not bypass !!!')
        # Execution:
        # 1 baselines
        # [TODO]
        
        # 2 orchestrated_attn_func:
        # [TODO]: check corretness of da_config✅&exp_configs✅
        benchmark_op = partial(benchmark_orchestrate_bsa,
            args, orchestrated_attn_func, da_config, exp_config, tensor_buf,
            inter_bsa_execution_plans=inter_bsa_execution_plans, inter_comp_plans_dicts=inter_comp_plans_dicts,
            global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
            warmup=WARMUP, num_iter=NUM_ITER, log=True,
        )
        bench_results = benchmark_op(use_cudagraph=False)   # List[[Tflops/s, s]] for rank0, List[[None, None]] for others
        assert len(bench_results) == len(inter_comp_plans_dicts) == len(sim_times)
        
        # Save bench_results to profile json file
        if torch.distributed.get_rank() == 0:
            for key, bench_result, sim_time in zip(keys, bench_results, sim_times):
                assert key not in inter_bsa_exe_plans_profile, f'profile_all_inter_BSA is profiled by grained of all ablation tests !!!'
                # Add sim_time in bench_results
                bench_result['sim_time'] = f'{sim_time:.3e}'
                inter_bsa_exe_plans_profile[key] = bench_result
            with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'w') as f:
                json.dump(inter_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')

def profile_all_inter_full(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor, prof_db: Prof_DB):
    # key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # print_rank_0(f'key_preffix: {key_preffix}')

    PROC_INFO = get_global_var(f'PROC_INFO')
    # [TODO]: Support baseline here !!! @yqg
    # baseline_funcs = [
    #     ring_flash_attn_func,
    #     zigzag_ring_flash_attn_func,      # baseline
    # ]
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    rank = PROC_INFO['rank']
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'

    # Bypass useless node !!!
    hierarchy_cp = da_config.hierarchy_sp
    if hierarchy_cp > node_num:
        print_rank_0(f'[WARN]: Current task needs {hierarchy_cp} nodes, but now there are only {node_num} nodes !!!')
        return
    if node_id >= hierarchy_cp:
        return
    # Correct gloo_global_group&ncclcomm_global for current task
    sub_group_key = tuple(range(hierarchy_cp * local_size))
    assert rank in sub_group_key
    
    cpu_group_dict = get_global_var_orch('cpu_group_dict')
    if sub_group_key not in cpu_group_dict.keys():
        assert False
        cpu_group_dict[sub_group_key] = torch.distributed.new_group(sub_group_key, backend='gloo')
        set_global_var_orch('cpu_group_dict', cpu_group_dict)
    gloo_global_group = cpu_group_dict[sub_group_key]
    torch.distributed.barrier(group=gloo_global_group)
    
    ncclcomm_dict = get_global_var_orch('ncclcomm_dict')
    if sub_group_key not in ncclcomm_dict.keys():
        assert False
        ncclcomm_dict[sub_group_key] = PyNcclCommunicator(gloo_global_group, ranks=sub_group_key, device=torch.cuda.current_device())
        set_global_var_orch('ncclcomm_dict', ncclcomm_dict)
    ncclcomm_global = ncclcomm_dict[sub_group_key]
    # End
    
    # experiment variables
    WARMUP, NUM_ITER = 11, 20 # most, best performance for most cases
    WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    
    # Prepare database
    with open(prof_db.INTER_BSA_EXE_PLANS_KV, 'r') as f:
        inter_bsa_exe_plans_dict = json.load(f)
    with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    
    # Create keys
    plan_key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config=full'
    KERNEL_TILE_TYPEs = ['w/o_kernel_tile', 'w_kernel_tile'] if max(da_config.Nh) == 1 else ['w/o_kernel_tile']
    ablation_dicts = create_ablation_configs_for_full(hierarchy_cp, KERNEL_TILE_TYPEs=KERNEL_TILE_TYPEs)
    profile_key_prefix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    profile_key_suffixes = [f"_ablation=({ad['KERNEL_TILE_TYPE']},Y={ad['Y']},X={ad['X']},dim={ad['first_dim']})" \
        for ad in ablation_dicts]
    profile_keys = [f'{profile_key_prefix}{profile_key_suffix}' for profile_key_suffix in profile_key_suffixes]
    print_rank_0(f'plan_key_preffix: {plan_key_preffix}, profile_key_prefix: {profile_key_prefix}')
    
    # Generate zip(inter_bsa_execution_plans, inter_comp_plans_dicts, sim_times)
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    sim_times = []
    filtered_ablation_dicts = []
    filtered_profile_keys = []  # the above 5 lists share the same length
    for ad, profile_key in zip(ablation_dicts, profile_keys):
        if profile_key not in inter_bsa_exe_plans_profile.keys():
            filtered_ablation_dicts.append(ad)
            filtered_profile_keys.append(profile_key)
    
    for ad in filtered_ablation_dicts:
        # load/generate exe_plan
        if ad['KERNEL_TILE_TYPE'] == 'w/o_kernel_tile': # non-fused
            plan_key = f"{plan_key_preffix}_ablation=(Y={ad['Y']},X={ad['X']},dim={ad['first_dim']})"
            plan_id = inter_bsa_exe_plans_dict[plan_key]
            with open(f'{prof_db.INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
                inter_bsa_execution_plan: Execution_Plan = pickle.load(fin)
        else: # fused
            inter_bsa_execution_plan = Fused_Execution_Plan(ad['Y'], ad['X'], None, fob=exp_config.fob, m_config=prof_db.m_config)
        # generate inter_comp_plans_dict
        inter_comp_plans_dict = prepare_inter_comp_plans(inter_bsa_execution_plan, prof_db, da_config)
        # append to lists
        sim_times.append(- 0.0) # calc sim_time # [TODO]
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
    
    is_bypass = len(inter_bsa_execution_plans) == 0
    if not is_bypass:
        print_rank_0(f'Not bypass !!!')
        # Execution:
        # 1 baselines
        # [TODO]
        
        # 2 orchestrated_attn_func:
        # [TODO]: check corretness of da_config✅&exp_configs✅
        benchmark_op = partial(benchmark_orchestrate_bsa,
            args, orchestrated_attn_func, da_config, exp_config, tensor_buf,
            inter_bsa_execution_plans=inter_bsa_execution_plans, inter_comp_plans_dicts=inter_comp_plans_dicts,
            global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
            warmup=WARMUP, num_iter=NUM_ITER, log=True,
        )
        bench_results = benchmark_op(use_cudagraph=False)   # List[[Tflops/s, s]] for rank0, List[[None, None]] for others
        assert len(bench_results) == len(inter_comp_plans_dicts) == len(sim_times)
        
        # Save bench_results to profile json file
        if torch.distributed.get_rank() == 0:
            for key, bench_result, sim_time in zip(filtered_profile_keys, bench_results, sim_times):
                assert key not in inter_bsa_exe_plans_profile, f'profile_all_inter_BSA is profiled by grained of all ablation tests !!!'
                # Add sim_time in bench_results
                bench_result['sim_time'] = f'{sim_time:.3e}'
                inter_bsa_exe_plans_profile[key] = bench_result
            with open(prof_db.INTER_BSA_EXE_PLANS_PROFILE, 'w') as f:
                json.dump(inter_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')

def step4_profile_inter_bsa_exe_plans(inter_exp_da_configs, ncclcomm_global, gloo_global_group, prof_db, profile_func=profile_all_inter_BSA):
    MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    for exp_da_config in inter_exp_da_configs:
        da_config = exp_da_config['da_config']
        MAX_S_perG = max(MAX_S_perG, max(da_config.S_per_gpu))
        MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
        MAX_D = max(MAX_D, da_config.shape_config['D'])
        MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
    report_memory(f'Before tensor_buf allocated')
    BUF_ELE_NUM = ((MAX_bs * MAX_S_perG * MAX_NH * (MAX_D + 1)) * (4 * 8) * (4))
    BUF_ELE_NUM = ((MAX_bs * MAX_S_perG * MAX_NH * (MAX_D + 1)) * (4 * 8) * (2))
    SYNC_SIZE = get_global_var('SYNC_SIZE')
    tensor_buf = torch.empty(
        max(BUF_ELE_NUM, SYNC_SIZE // (torch.finfo(DTYPE).bits // 8)),
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    report_memory(f'After tensor_buf allocated')
    print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    args = parse_args()
    inter_plan_id = 0
    
    for exp_da_config in inter_exp_da_configs:
        exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
        profile_func(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf, prof_db)
        torch.distributed.barrier(gloo_global_group)
        inter_plan_id += 1

def step4_profile_inter_full_exe_plans(dummy_inter_exp_da_configs, shape_config_dict, ncclcomm_global, gloo_global_group, prof_db):
    # Create inter_exp_da_full_configs
    inter_exp_da_configs: List[dict] = []
    inter_node_shape_configs = shape_config_dict['inter']
    for exp_da_config in dummy_inter_exp_da_configs:
        exp_config = exp_da_config['exp_config']
        dummy_da_config: Dist_Attn_Config = exp_da_config['da_config']  # determine (fob, CPs)
        # print_rank_0(f'dummy_da_config: {dummy_da_config}')
        # vary shapes: (Nh, S)
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
                            dummy_da_config.bsa_config, 
                            shape_config=shape_config,
                            hierarchy=dummy_da_config.hierarchy,
                        )
                        S_per_gpu = S_tot // da_config.tot_sp
                        if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                            continue
                        inter_exp_da_configs.append({
                            'exp_config': exp_config,
                            'da_config': da_config,
                        })
    step4_profile_inter_bsa_exe_plans(inter_exp_da_configs, ncclcomm_global, gloo_global_group, prof_db, profile_func=profile_all_inter_full)
    # # Calc tensor_buf size
    # MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    # for exp_da_config in inter_exp_da_configs:
    #     da_config: Dist_Attn_Config = exp_da_config['da_config']
    #     MAX_S_perG = max(MAX_S_perG, max(da_config.S_per_gpu))
    #     MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
    #     MAX_D = max(MAX_D, da_config.shape_config['D'])
    #     MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
    # report_memory(f'Before tensor_buf allocated')
    # BUF_ELE_NUM = ((MAX_bs * MAX_S_perG * MAX_NH * (MAX_D + 1)) * (4 * 8) * (4))
    # BUF_ELE_NUM = ((MAX_bs * MAX_S_perG * MAX_NH * (MAX_D + 1)) * (4 * 8) * (2))
    # SYNC_SIZE = get_global_var('SYNC_SIZE')
    # tensor_buf = torch.empty(
    #     max(BUF_ELE_NUM, SYNC_SIZE // (torch.finfo(DTYPE).bits // 8)),
    #     device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    # )   # 6 * 512MB = 3GB
    # report_memory(f'After tensor_buf allocated')
    # print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    # print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    # args = parse_args()
    
    # # Execute plans
    # inter_plan_id = 0
    # for exp_da_config in inter_exp_da_configs:
    #     exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
    #     profile_all_inter_full(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf, prof_db)
    #     torch.distributed.barrier(gloo_global_group)
    #     inter_plan_id += 1

def main(args):
    set_global_var('ARGS', args)
    # Initialize distribution
    ncclcomm_global, gloo_global_group = initialize_distribution()
    # Initialize Profile_DataBase
    prof_db = initialize_prof_db(gloo_global_group)

    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict, exp_configs = step0_top_down_decompose()    
    from search_algo.task1_bsa import step3_generate_inter_bsa_exe_plans, step3_generate_inter_dense_exe_plans
    
    if args.exp_class == 'dense_train': # full, causal
        intra_exp_da_full_configs, inter_exp_da_causal_configs = \
            step3_generate_inter_dense_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db, is_bypass_mode=True) # Full + Causal
        # Step4: Profile all dense at inter_SP; multiple nodes, one processor occupies one gpu and even cpus; (w cache/bypass)
        if torch.cuda.is_available():
            step4_profile_inter_full_exe_plans(intra_exp_da_full_configs, shape_config_dict, ncclcomm_global, gloo_global_group, prof_db)
            step4_profile_inter_bsa_exe_plans(inter_exp_da_causal_configs, ncclcomm_global, gloo_global_group, prof_db)
    else:
        inter_exp_da_configs = step3_generate_inter_bsa_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db, is_bypass_mode=True)  # Bypass mode
        # Step4: Profile all BSA at inter_SP; multiple nodes, one processor occupies one gpu and even cpus; (w cache/bypass)
        if torch.cuda.is_available():
            step4_profile_inter_bsa_exe_plans(inter_exp_da_configs, ncclcomm_global, gloo_global_group, prof_db)
    
if __name__ == '__main__':
    main(parse_args())