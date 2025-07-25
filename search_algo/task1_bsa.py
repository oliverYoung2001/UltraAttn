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
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs, bsa_is_dense, bsa_is_causal
from search_algo.dense_utils import create_plan_for_full, split_dense_configs, create_plan_key_suffixes_for_full, \
    create_ablation_configs_for_full, CP2ParD_map
from search_algo.utils import combine_list_to_0, convert_block_table_to_value, parse_args, print_rank_0
from search_algo.database import Prof_DB
import torch
import torch.distributed as dist
from functools import partial
import socket
import tests
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from search_algo.workload_partition import solve_sparse_from_bsa, naive_allocate_strategy
from search_algo.benchmark import benchmark_orchestrate_bsa
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func
import json
import random
from typing import List, Tuple, Union, Optional
from search_algo.exp_configs import step0_top_down_decompose
from search_algo.initialize import initialize_distribution, initialize_prof_db
from orchestrated_attn.global_vars import set_global_var as set_global_var_orch
from orchestrated_attn.global_vars import get_global_var as get_global_var_orch
from search_algo.utils import Block_Type
import math

# In-file global vars
DTYPE = torch.bfloat16
# End

def get_general_bsa_cc_optimal_schedule_old(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB) -> Dist_Attn_Config:
    # [NOTE]: Is general, i.g. is intra func useless ???
    fob = exp_config.fob
    hierarchy = da_config.hierarchy # (0, 1) -> (inter, intra)
    hier_pre = 'inter' if hierarchy == 0 else 'intra'
    CP = da_config.bsa_config.CP    # (intra, inter)
    
    GENERAL_BSA_ALLOCATION = prof_db.INTRA_BSA_ALLOCATION if hierarchy else prof_db.INTER_BSA_ALLOCATION
    key = f'fob={exp_config.fob}_bsa_config={{{da_config.bsa_config}}}'  # [TODO]
    print_rank_0(f'{hier_pre}_bsa_allocation_key: {key}')
    with open(f'{GENERAL_BSA_ALLOCATION}', 'r') as f:
        general_bsa_allocation_dict = json.load(f)
    if key in general_bsa_allocation_dict.keys():
        print_rank_0(f'Bypassed !!!')
        value = general_bsa_allocation_dict[key]
        schedule_table = np.array(value['schedule_table'], dtype=np.int32)
        assert value['Par_D'] == schedule_table.shape[-1]
        schedule_results = {
            'CP': CP,   # (intra, inter)
            # 'cmap': da_config.bsa_config.cmap,
            'table': schedule_table,
        }
        # print_rank_0(f'cmap: {da_config.bsa_config.cmap}') # None !!!
    else:
        print_rank_0(f'Not bypass !!!')
        # assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
        # Par_D = max(CP[hierarchy], da_config.bsa_config.block_table.shape[0])
        Par_D = None if hierarchy else CP[not hierarchy]
        # print_rank_0(f'Par_D: {Par_D}, CP[not hierarchy]: {CP[not hierarchy]}')
        if hierarchy == 0: # inter; [TODO]
            assert Par_D == CP[not hierarchy], f'Inter bsa schedule not support Par_D={Par_D} > CP[0]={CP[0]} now.'
            
        schedule_results = solve_sparse_from_bsa(da_config.bsa_config, fob=fob, ParD=Par_D, hierarchy=hierarchy)  # modify here !!!✅
        schedule_table = schedule_results['table']
        # print_rank_0(f'schedule_table: {schedule_table.dtype}')    # int32
        value = {
            'Par_D': schedule_table.shape[-1],
            'schedule_table': schedule_table.tolist(),
        }
        general_bsa_allocation_dict[key] = value
        with open(f'{GENERAL_BSA_ALLOCATION}', 'w') as f:
            json.dump(general_bsa_allocation_dict, f)
    
    cc_optimal_schedule = get_cc_optimal_schedule_from_table(da_config, prof_db.m_config, schedule_results)
    
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print_rank_0(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
    return cc_optimal_schedule
   
def get_general_bsa_cc_optimal_schedule(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB, gpu_tile_type: Optional[str] = None) -> Dist_Attn_Config:
    # gpu_tile_type: None -> w_tile; 'naive' -> 'naive'; 'comp_balance' -> 'comp_balance'
    # [NOTE]: Is general, i.g. is intra func useless ???
    fob = exp_config.fob
    hierarchy = da_config.hierarchy # (0, 1) -> (inter, intra)
    hier_pre = 'inter' if hierarchy == 0 else 'intra'
    CP = da_config.bsa_config.CP    # (intra, inter)
    
    GENERAL_BSA_ALLOCATION = prof_db.INTRA_BSA_ALLOCATION if hierarchy else prof_db.INTER_BSA_ALLOCATION
    key_suffix = '' if gpu_tile_type is None else f'_{gpu_tile_type}'
    key = f'fob={exp_config.fob}_bsa_config={{{da_config.bsa_config}}}{key_suffix}'
    print_rank_0(f'{hier_pre}_bsa_allocation_key: {key}')
    with open(f'{GENERAL_BSA_ALLOCATION}', 'r') as f:
        general_bsa_allocation_dict = json.load(f)
    if key in general_bsa_allocation_dict.keys():
        print_rank_0(f'Bypassed !!!')
        value = general_bsa_allocation_dict[key]
        schedule_table = np.array(value['schedule_table'], dtype=np.int32)
        assert value['Par_D'] == schedule_table.shape[-1]
        schedule_results = {
            'CP': CP,   # (intra, inter)
            # 'cmap': da_config.bsa_config.cmap,
            'table': schedule_table,
        }
        # print_rank_0(f'cmap: {da_config.bsa_config.cmap}') # None !!!
    else:
        print_rank_0(f'Not bypass !!!')
        # assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
        # Par_D = max(CP[hierarchy], da_config.bsa_config.block_table.shape[0])
        if hierarchy == 0 and bsa_is_causal(da_config.bsa_config):  # Manually set Par_D for causal at node level
            Par_D = CP2ParD_map[CP[not hierarchy]]
        else:
            Par_D = None if hierarchy else CP[not hierarchy]
        # print_rank_0(f'Par_D: {Par_D}, CP[not hierarchy]: {CP[not hierarchy]}, is_causal: {bsa_is_causal(da_config.bsa_config)}')
        if hierarchy == 0 and not bsa_is_causal(da_config.bsa_config): # inter; [TODO]
            assert Par_D == CP[not hierarchy], f'Inter bsa schedule not support Par_D={Par_D} > CP[0]={CP[0]} for general bsa patterns.'
        # schedule_results: { 'Par_D': xxx, 'cmap': xxx, 'table': xxx}
        if gpu_tile_type is None:
            schedule_results = solve_sparse_from_bsa(da_config.bsa_config, fob=fob, ParD=Par_D, hierarchy=hierarchy)  # modify here !!!✅
        elif gpu_tile_type == 'naive':
            schedule_results = naive_allocate_strategy(da_config.bsa_config, fob=fob, ParD=Par_D, hierarchy=hierarchy)
        elif gpu_tile_type == 'comp_balance':
            pass    # [TODO]
        schedule_table = schedule_results['table']
        # print_rank_0(f'schedule_table: {schedule_table.dtype}')    # int32
        value = {
            'Par_D': schedule_table.shape[-1],
            'schedule_table': schedule_table.tolist(),
        }
        general_bsa_allocation_dict[key] = value
        with open(f'{GENERAL_BSA_ALLOCATION}', 'w') as f:
            json.dump(general_bsa_allocation_dict, f)
    
    cc_optimal_schedule = get_cc_optimal_schedule_from_table(da_config, prof_db.m_config, schedule_results)
    
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print_rank_0(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
    return cc_optimal_schedule
    
def generate_intra_bsa_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB, is_bypass_mode: bool):
    exp_config.hierarchy = da_config.hierarchy = 1
    m_config = prof_db.m_config
    print_rank_0(f'da_config.shape_config: {da_config.shape_config}')
    cc_optimal_schedule = get_general_bsa_cc_optimal_schedule(exp_config, da_config, prof_db)
    cc_naive_schedule = get_general_bsa_cc_optimal_schedule(exp_config, da_config, prof_db, gpu_tile_type='naive') # w/o gpu tile
    # exit(0)
    
    # Generate Intra_Execution_Plans:
    intra_bsa_exe_plans_dict_changed = False
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    
    def not_bypass_behavior(d_graph: Dependent_Graph, KERNEL_TILE_TYPE: str, KERNEL_SCHEDULE_TYPE: str):
        assert not is_bypass_mode, f'All in generate_inter_bsa_execution_plans must be bypassed in bypass mode !!!'
        print_rank_0(f'Not bypass !!!')
        if KERNEL_TILE_TYPE == 'w_kernel_tile':
            gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
            execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=plan_type)
        elif KERNEL_TILE_TYPE == 'w/o_kernel_tile':
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=KERNEL_SCHEDULE_TYPE)
        else:
            assert False, f'[ERROR]: Unknown KERNEL_TILE_TYPE={KERNEL_TILE_TYPE}'
        execute_plan.print_lp_result()
        # Dump Execution_Plan:
        plan_id = max(intra_bsa_exe_plans_dict.values()) + 1 if intra_bsa_exe_plans_dict else 0
        intra_bsa_exe_plans_dict[key] = plan_id
        plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
        nonlocal intra_bsa_exe_plans_dict_changed
        intra_bsa_exe_plans_dict_changed = True
        
    def bypass_behavior(key: str):
        # print_rank_0(f'Bypassed !!!')
        plan_id = intra_bsa_exe_plans_dict[key]
        plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'rb') as fin:
            execute_plan: Execution_Plan = pickle.load(fin)
        # execute_plan.print_lp_result()
        print_rank_0(f'end_time={execute_plan.get_end_time():.3e}')

    #   1. Generate 1 type, i.e. w/o_gpu_tile Execution_Plan for ablations:
    d_graph = Dependent_Graph(cc_naive_schedule, exp_config.fob)
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    GPU_TILE_TYPE = 'w/o_gpu_tile'
    KERNEL_TILE_TYPE = 'w/o_kernel_tile' 
    KERNEL_SCHEDULE_TYPE = 'Flexflow'
    key_suffix = f'_ablation=({GPU_TILE_TYPE},{KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
    key = f'{key_preffix}{key_suffix}'
    print_rank_0(f'intra_bsa_exe_plan_key: {key}')
    if key not in intra_bsa_exe_plans_dict.keys():
        not_bypass_behavior(d_graph, KERNEL_TILE_TYPE, KERNEL_SCHEDULE_TYPE)
    else:
        bypass_behavior(key)
    
    #   2. Generate 4=2x2 types of Execution_Plans for ablations:
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    plan_types = ['Flexflow', 'ILP']
    for plan_type in plan_types:
        # KERNEL_SCHEDULE_TYPE = "ILP" if plan_type == "automatic" else "Flexflow"
        KERNEL_SCHEDULE_TYPE = plan_type
        # w/o Kernel Tile Execution_Plan:
        KERNEL_TILE_TYPE = 'w/o_kernel_tile'
        print_rank_0(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:')
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print_rank_0(f'intra_bsa_exe_plan_key: {key}')
        if key not in intra_bsa_exe_plans_dict.keys():
            not_bypass_behavior(d_graph, KERNEL_TILE_TYPE, KERNEL_SCHEDULE_TYPE)
        else:
            bypass_behavior(key)
        
        # w Kernel Tile Execution_Plans:
        KERNEL_TILE_TYPE = 'w_kernel_tile'
        print_rank_0(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:')
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print_rank_0(f'intra_bsa_exe_plan_key: {key}')
        if key not in intra_bsa_exe_plans_dict.keys():
            not_bypass_behavior(d_graph, KERNEL_TILE_TYPE, KERNEL_SCHEDULE_TYPE)
        else:
            bypass_behavior(key)
    
    if intra_bsa_exe_plans_dict_changed:
        # assert not torch.cuda.is_available(), f'intra_bsa_exe_plans_dict should not be changed in GPU nodes'
        with open(f'{prof_db.INTRA_BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump(intra_bsa_exe_plans_dict, f)

def profile_all_intra_BSA(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor, prof_db: Prof_DB):
    PROC_INFO = get_global_var(f'PROC_INFO')
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    local_rank = PROC_INFO['local_rank']
    node_num = PROC_INFO['node_num']
    node_id = PROC_INFO['nodeid']
    rank = PROC_INFO['rank']
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'
    
    # Bypass useless node !!!
    hierarchy_cp = da_config.hierarchy_sp
    if hierarchy_cp > local_size:
        print_rank_0(f'[WARN]: Current task needs {hierarchy_cp} gpus, but now there are only {local_size} gpus !!!')
        return
    if local_rank >= hierarchy_cp:
        return
    # Correct gloo_global_group&ncclcomm_global for current task
    sub_group_key = tuple(range(hierarchy_cp))
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
    # WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    # WARMUP, NUM_ITER = 1, 1
    
    # Prepare database
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)
    # Create keys
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    key_suffixes = [f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)']
    key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                        for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
    print_rank_0(f'key_preffix: {key_preffix}')
    # Generate inter_comp_plans_dicts
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    sim_times = []
    for key in keys:
        if key in intra_bsa_exe_plans_profile.keys():   # Bypassed
            continue
        # Create dummy inter_bsa_execution_plan
        inter_bsa_execution_plan: Optional[Execution_Plan] = Execution_Plan.create_one_node_exe_plan(exp_config.fob)
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        # load exe_plan
        plan_id = intra_bsa_exe_plans_dict[key]
        with open(f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
            intra_bsa_execution_plan: Execution_Plan = pickle.load(fin)
        sim_times.append(intra_bsa_execution_plan.end_time)
        # OBJ1: build inter_comp_plans_dict
        # OBJ2: Set correct execution_plan to each inter kernel
        # inter_comp_plans_dict: {intra_bsa_key: intra_bsa_exe_plan}; intra_bsa_key only aims to deduplicate
        inter_comp_plans_dict = {
            ((1, 1), str(da_config.bsa_config.bsa_repr)): intra_bsa_execution_plan,
        }
        only_kernel = inter_bsa_execution_plan.gpu_kernel_lists[node_id][0]
        assert len(inter_comp_plans_dict.values()) == 1, "Only 1 intra execution plan is needed for profiling"
        only_kernel.execution_plan = list(inter_comp_plans_dict.values())[0]
        
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
        
    # is_bypass = all([key in intra_bsa_exe_plans_profile.keys() for key in keys])
    is_bypass = len(inter_bsa_execution_plans) == 0
    if not is_bypass:
        print_rank_0(f'Not bypass !!!')
        # Execution:
        # 1 baselines
        # [TODO]
        
        # 2 orchestrated_attn_func:
        # [TODO]: check corretness of da_config✅&exp_configs✅
        # [TODO]: adapt to modified benchmark_orchestrate_bsa !!!
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
                assert key not in intra_bsa_exe_plans_profile, f'profile_all_intra_BSA is profiled by grained of all ablation tests !!!'
                # Add sim_time in bench_results
                bench_result['sim_time'] = f'{sim_time:.3e}'
                intra_bsa_exe_plans_profile[key] = bench_result
            with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'w') as f:
                json.dump(intra_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')

def profile_all_intra_full(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor, prof_db: Prof_DB):
    PROC_INFO = get_global_var(f'PROC_INFO')
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    local_rank = PROC_INFO['local_rank']
    node_num = PROC_INFO['node_num']
    node_id = PROC_INFO['nodeid']
    rank = PROC_INFO['rank']
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'
    
    # Bypass useless node !!!
    hierarchy_cp = da_config.hierarchy_sp
    if hierarchy_cp > local_size:
        print_rank_0(f'[WARN]: Current task needs {hierarchy_cp} gpus, but now there are only {local_size} gpus !!!')
        return
    if local_rank >= hierarchy_cp:
        return
    # Correct gloo_global_group&ncclcomm_global for current task
    sub_group_key = tuple(range(hierarchy_cp))
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
    # WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    # WARMUP, NUM_ITER = 1, 1
    
    # Prepare database
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)
    # Create keys   # TODO
    # key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # key_suffixes = [f'_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)']
    # key_suffixes += [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
    #                     for KERNEL_SCHEDULE_TYPE in ['Flexflow', 'ILP'] \
    #                         for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    # keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
    
    plan_key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config=full'
    KERNEL_TILE_TYPEs = ['w/o_kernel_tile', 'w_kernel_tile'] if max(da_config.Nh) == 1 else ['w/o_kernel_tile']
    ablation_dicts = create_ablation_configs_for_full(hierarchy_cp, KERNEL_TILE_TYPEs=KERNEL_TILE_TYPEs)
    profile_key_prefix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    profile_key_suffixes = [f"_ablation=({ad['KERNEL_TILE_TYPE']},Y={ad['Y']},X={ad['X']},dim={ad['first_dim']})" \
        for ad in ablation_dicts]
    profile_keys = [f'{profile_key_prefix}{profile_key_suffix}' for profile_key_suffix in profile_key_suffixes]
    print_rank_0(f'plan_key_preffix: {plan_key_preffix}, profile_key_prefix: {profile_key_prefix}')
    
    # Generate inter_comp_plans_dicts
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    sim_times = []
    filtered_ablation_dicts = []
    filtered_profile_keys = []  # the above 5 lists share the same length
    for ad, profile_key in zip(ablation_dicts, profile_keys):
        if profile_key not in intra_bsa_exe_plans_profile.keys():
            filtered_ablation_dicts.append(ad)
            filtered_profile_keys.append(profile_key)
    
    for ad in filtered_ablation_dicts:
        # Create dummy inter_bsa_execution_plan
        inter_bsa_execution_plan: Optional[Execution_Plan] = Execution_Plan.create_one_node_exe_plan(exp_config.fob)
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        if ad['KERNEL_TILE_TYPE'] == 'w/o_kernel_tile':
            # get exe_plan
            plan_key = f"{plan_key_preffix}_ablation=(Y={ad['Y']},X={ad['X']},dim={ad['first_dim']})"
            plan_id = intra_bsa_exe_plans_dict[plan_key]
            with open(f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
                intra_bsa_execution_plan: Execution_Plan = pickle.load(fin)
            # [TODO]: calc sim_time: update every real_comp_map_key and comm_raw_map_key according new S and Nh
        else:   # w_kernel_tile
            intra_bsa_execution_plan = Fused_Execution_Plan(ad['Y'], ad['X'], None, fob=exp_config.fob, m_config=prof_db.m_config)
            # [TODO]: calc sim_time: How to do it ?
        # calc sim_time # [TODO]
        sim_times.append(- 0.0)
        
        # OBJ1: build inter_comp_plans_dict
        # OBJ2: Set correct execution_plan to each inter kernel
        # inter_comp_plans_dict: {intra_bsa_key: intra_bsa_exe_plan}; intra_bsa_key only aims to deduplicate
        inter_comp_plans_dict = {
            ((1, 1), str(da_config.bsa_config.bsa_repr)): intra_bsa_execution_plan,
        }
        only_kernel = inter_bsa_execution_plan.gpu_kernel_lists[node_id][0]
        assert len(inter_comp_plans_dict.values()) == 1, "Only 1 intra execution plan is needed for profiling"
        only_kernel.execution_plan = list(inter_comp_plans_dict.values())[0]
        
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
        
    # is_bypass = all([key in intra_bsa_exe_plans_profile.keys() for key in keys])
    is_bypass = len(inter_bsa_execution_plans) == 0
    if not is_bypass:
        print_rank_0(f'Not bypass !!!')
        # Execution:
        # 1 baselines
        # [TODO]
        
        # 2 orchestrated_attn_func:
        # [TODO]: check correctness of da_config✅&exp_configs✅
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
                assert key not in intra_bsa_exe_plans_profile, f'profile_all_intra_BSA is profiled by grained of all ablation tests !!!'
                # Add sim_time in bench_results
                bench_result['sim_time'] = f'{sim_time:.3e}'
                intra_bsa_exe_plans_profile[key] = bench_result
            with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'w') as f:
                json.dump(intra_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')

def generate_inter_bsa_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB, is_bypass_mode: bool): # ✅
    exp_config.hierarchy = da_config.hierarchy = 0  # Inter
    
    m_config = prof_db.m_config
    print_rank_0(f'da_config.shape_config Inter: {da_config.shape_config}')
    
    # Calc optimal schedule
    cc_optimal_schedule = get_general_bsa_cc_optimal_schedule(exp_config, da_config, prof_db)
    # exit(0)
    # Generate Inter_Execution_Plans:
    inter_bsa_exe_plans_dict_changed = False
    with open(prof_db.INTER_BSA_EXE_PLANS_KV, 'r') as f:
        inter_bsa_exe_plans_dict = json.load(f)
    
    def not_bypass_behavior(bsa_comp_key_suffixes: List[str], KERNEL_SCHEDULE_TYPE: str):
        assert not is_bypass_mode, f'All in generate_inter_bsa_execution_plans must be bypassed in bypass mode !!!'
        print_rank_0(f'Not bypass !!!')
        #   1. Build Dependent_Graph
        d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob, is_inter_bsa=True, bsa_comp_key_suffixes=bsa_comp_key_suffixes)
        # [TODO]: different d_graph regarding to ablations !!! But we use ('ILP', 'w kernel tile') currently.
        #   2. Generate Execution_Plan
        if KERNEL_TILE_TYPE == 'w_kernel_tile' and bsa_is_dense(da_config.bsa_config):  # For causal, with kernel-tile
            gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
            execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=KERNEL_SCHEDULE_TYPE)
        else:
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=KERNEL_SCHEDULE_TYPE) # [TODO]: m_config is error !!!
        execute_plan.print_lp_result()
        # Dump Execution_Plan:
        plan_id = max(inter_bsa_exe_plans_dict.values()) + 1 if inter_bsa_exe_plans_dict else 0
        inter_bsa_exe_plans_dict[key] = plan_id
        plan_file = f'{prof_db.INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
        nonlocal inter_bsa_exe_plans_dict_changed
        inter_bsa_exe_plans_dict_changed = True
                    
    def bypass_behavior(key):
        # print_rank_0(f'Bypassed !!!')
        plan_id = inter_bsa_exe_plans_dict[key]
        plan_file = f'{prof_db.INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'rb') as fin:
            execute_plan: Execution_Plan = pickle.load(fin)
        # execute_plan.print_lp_result()
        print_rank_0(f'end_time={execute_plan.get_end_time():.3e}')
    
    #   1. Generate 1 type, i.e. w/o_gpu_tile Execution_Plan for ablations:
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    GPU_TILE_TYPE = 'w/o_gpu_tile'
    KERNEL_TILE_TYPE = 'w/o_kernel_tile' 
    KERNEL_SCHEDULE_TYPE = 'Flexflow'
    key_suffix = f'_ablation=({GPU_TILE_TYPE},{KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
    key = f'{key_preffix}{key_suffix}'
    print_rank_0(f'inter_bsa_exe_plan_key: {key}')
    if key not in inter_bsa_exe_plans_dict.keys():
        bsa_comp_key_suffixes = [key_suffix]
        not_bypass_behavior(bsa_comp_key_suffixes, KERNEL_SCHEDULE_TYPE)
    else:
        bypass_behavior(key)
    
    #   2. Generate 4=2x2 types of Execution_Plans for ablations:
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    plan_types = ['Flexflow', 'ILP']
    kernel_tile_types = ['w/o_kernel_tile', 'w_kernel_tile']
    for i, KERNEL_SCHEDULE_TYPE in enumerate(plan_types):
        for j, KERNEL_TILE_TYPE in enumerate(kernel_tile_types):
            print_rank_0(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:')
            key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
            key = f'{key_preffix}{key_suffix}'
            print_rank_0(f'inter_bsa_exe_plan_key: {key}')
            if key not in inter_bsa_exe_plans_dict.keys():
                bsa_comp_key_suffixes = [f'_ablation=({kernel_tile_types[l]},{plan_types[k]})' for k in range(i + 1) for l in range(j + 1)]
                not_bypass_behavior(bsa_comp_key_suffixes, KERNEL_SCHEDULE_TYPE)
            else:
                bypass_behavior(key)
    
    if inter_bsa_exe_plans_dict_changed:
        # assert not torch.cuda.is_available(), f'inter_bsa_exe_plans_dict should not be changed in GPU nodes'
        with open(prof_db.INTER_BSA_EXE_PLANS_KV, 'w') as f:
            json.dump(inter_bsa_exe_plans_dict, f)

def step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict: dict, exp_configs, prof_db, is_bypass_mode: bool = False):
    # Step1: Generate the intra-BSA; need all cpus on one node; (w cache/bypass)
    intra_node_shape_configs = shape_config_dict['intra']
    intra_plan_id = 0
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
                                hierarchy=1,
                            )
                            S_per_gpu = S_per_node // da_config.hierarchy_sp
                            if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                                continue
                            intra_exp_da_configs.append({
                                'exp_config': exp_config,
                                'da_config': da_config,
                            })
                            print_rank_0(f'intra_plan_id: {intra_plan_id}')
                            generate_intra_bsa_execution_plans(exp_config, da_config, prof_db, is_bypass_mode)
                            intra_plan_id += 1
    return intra_exp_da_configs

def generate_general_full_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, hierarchy: bool,
                                          prof_db: Prof_DB, is_bypass_mode: bool):
    # Exe_plans for full is only for w/o kernel-tile and contain only graph structure without sim_time.
    exp_config.hierarchy = da_config.hierarchy = hierarchy  # (0, 1) -> (inter, intra)
    hierarchy_cp = da_config.hierarchy_sp
    if hierarchy == 0:
        BSA_EXE_PLANS_KV = prof_db.INTER_BSA_EXE_PLANS_KV
        BSA_EXE_PLANS_DIR = prof_db.INTER_BSA_EXE_PLANS_DIR
    else:
        BSA_EXE_PLANS_KV = prof_db.INTRA_BSA_EXE_PLANS_KV
        BSA_EXE_PLANS_DIR = prof_db.INTRA_BSA_EXE_PLANS_DIR
    
    # Generate Execution_Plans:
    bsa_exe_plans_dict_changed = False
    with open(BSA_EXE_PLANS_KV, 'r') as f:
        bsa_exe_plans_dict = json.load(f)
    
    def not_bypass_behavior(key: str, Y: int, X: int, first_dim: int):
        assert not is_bypass_mode, f'All in generate_general_full_execution_plans must be bypassed in bypass mode !!!'
        print_rank_0(f'Not bypass !!!')
        
        execute_plan = create_plan_for_full(da_config, prof_db.m_config, X, fob=exp_config.fob, first_dim=0)
        execute_plan.print_lp_result()
        # Dump Execution_Plan:
        plan_id = max(bsa_exe_plans_dict.values()) + 1 if bsa_exe_plans_dict else 0
        bsa_exe_plans_dict[key] = plan_id
        plan_file = f'{BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
        nonlocal bsa_exe_plans_dict_changed
        bsa_exe_plans_dict_changed = True
        
    def bypass_behavior(key: str):
        # print_rank_0(f'Bypassed !!!')
        plan_id = bsa_exe_plans_dict[key]
        plan_file = f'{BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
        with open(plan_file, 'rb') as fin:
            execute_plan: Execution_Plan = pickle.load(fin)
        # execute_plan.print_lp_result()
        print_rank_0(f'end_time={execute_plan.get_end_time():.3e}')

    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config=full'
    ablation_dicts = create_ablation_configs_for_full(hierarchy_cp)
    plan_key_suffixes = create_plan_key_suffixes_for_full(hierarchy_cp)
    # print_rank_0(f'ablation_dicts: {ablation_dicts}')
    for ad, plan_key_suffix in zip(ablation_dicts, plan_key_suffixes):
        plan_key = f'{key_preffix}{plan_key_suffix}'
        print_rank_0(f"{'inter' if hierarchy == 0 else 'intra'}_bsa_exe_plan_key: {plan_key}")
        if plan_key not in bsa_exe_plans_dict.keys():
            not_bypass_behavior(plan_key, ad['Y'], ad['X'], ad['first_dim'])
        else:
            bypass_behavior(plan_key)
    
    if bsa_exe_plans_dict_changed:
        with open(f'{BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump(bsa_exe_plans_dict, f)

def step1_generate_intra_full_exe_plans(intra_node_bsa_configs, shape_config_dict: dict, exp_configs, prof_db, is_bypass_mode: bool = False):
    hierarchy = 1
    # 1. For full, origin feature is implemented in `scripts/schedule/intra_attn_gen.sh` ✅
    dummy_shape_config = {
        'Nh': (1, 1),
        'S': (512, 512),    # total_S
        'bs': 1,
        'D': 128,
    }
    intra_exp_da_configs: List[dict] = []
    for exp_config in exp_configs:  # fobs
        for intra_node_bsa_config in intra_node_bsa_configs:    # CPs
            if not (intra_node_bsa_config.bsa_repr.block_table_raw.shape == (1, 1) \
                and intra_node_bsa_config.bsa_repr.block_table_raw[0][0].value == Block_Type.FULL.value):
                assert False
            da_config = Dist_Attn_Config.from_bsa_config(
                intra_node_bsa_config, 
                shape_config=dummy_shape_config,
                hierarchy=hierarchy,
            )
            intra_exp_da_configs.append({
                'exp_config': exp_config,
                'da_config': da_config,
            })
            generate_general_full_execution_plans(exp_config, da_config, hierarchy, prof_db, is_bypass_mode)
    return intra_exp_da_configs

def step2_profile_intra_bsa_exe_plans(intra_exp_da_configs, ncclcomm_global, gloo_global_group, prof_db):
    MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    for exp_da_config in intra_exp_da_configs:
        da_config = exp_da_config['da_config']
        MAX_S_perG = max(MAX_S_perG, max(da_config.S_per_gpu))
        MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
        MAX_D = max(MAX_D, da_config.shape_config['D'])
        MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
    BUF_ELE_NUM = (MAX_bs * MAX_S_perG * MAX_NH * MAX_D * 4) * 3                       # k, v, dk, dv
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 2) + (1 * (2 + 1))) * 2     # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
    SYNC_SIZE = get_global_var('SYNC_SIZE')
    tensor_buf = torch.empty(
        max(BUF_ELE_NUM, SYNC_SIZE // (torch.finfo(DTYPE).bits // 8)),
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    args = parse_args()
    for exp_da_config in intra_exp_da_configs:
        exp_config, da_config = exp_da_config['exp_config'], exp_da_config['da_config']
        profile_all_intra_BSA(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf, prof_db)
        torch.distributed.barrier(gloo_global_group)

def step2_profile_intra_full_exe_plans(intra_exp_da_configs, shape_config_dict: dict, ncclcomm_global, gloo_global_group, prof_db):
    MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    MAX_S_base_perG, MIN_S_base_perG = 0, sys.maxsize
    MAX_CP_INTER = 8    # [NOTE]: Assume that CP_inter <= 8 (HARDCODE)
    multiplying_powers = list(range(1, MAX_CP_INTER))
    intra_node_shape_configs = shape_config_dict['intra']
    for exp_da_config in intra_exp_da_configs:
        da_config: Dist_Attn_Config = exp_da_config['da_config']
        for Nh in intra_node_shape_configs['Nhs']:
            for S_per_node in intra_node_shape_configs['Ss']:
                for bs in intra_node_shape_configs['BSs']:
                    for D in intra_node_shape_configs['Ds']:
                        MAX_S_perG = max(MAX_S_perG, S_per_node // da_config.hierarchy_sp * max(multiplying_powers))
                        MAX_S_base_perG = max(MAX_S_base_perG, S_per_node // da_config.hierarchy_sp)
                        MIN_S_base_perG = min(MIN_S_base_perG, S_per_node // da_config.hierarchy_sp)
                        MAX_NH = max(MAX_NH, Nh)
                        MAX_D = max(MAX_D, D)
                        MAX_bs = max(MAX_bs, bs)
    MAX_S_perG = min(MAX_S_perG, shape_config_dict['S_per_gpu_BOUND'][1])
    
    BUF_ELE_NUM = (MAX_bs * MAX_S_perG * MAX_NH * MAX_D * 4) * 3                       # k, v, dk, dv
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 2) + (1 * (2 + 1))) * 2     # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
    SYNC_SIZE = get_global_var('SYNC_SIZE')
    tensor_buf = torch.empty(
        max(BUF_ELE_NUM, SYNC_SIZE // (torch.finfo(DTYPE).bits // 8)),
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_S_base_perG={MAX_S_base_perG}; MIN_S_base_perG={MIN_S_base_perG}; '
                 f'MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    args = parse_args()
    
    S_base = [1 << logS for logS in range(int(math.log2(MIN_S_base_perG)), int(math.log2(MAX_S_base_perG)) + 1)]
    Sqkvs = [S * power for S in S_base for power in multiplying_powers if S * power <= shape_config_dict['S_per_gpu_BOUND'][1]]
    Sqkvs = sorted(list(set(Sqkvs)))    # Sq per GPU
    Sqs = Skvs = Sqkvs
    
    for exp_da_config in intra_exp_da_configs:
        exp_config, dummy_da_config = exp_da_config['exp_config'], exp_da_config['da_config'] # fob, CP
        hierarchy_cp = dummy_da_config.hierarchy_sp
        for Nh in intra_node_shape_configs['Nhs']:
            for Sq in Sqs: # S per GPU
                for Skv in Skvs: # S per GPU
                    shape_config = {
                        'Nh': (Nh, Nh), # Different from dummy_da_config
                        'S': (Sq * hierarchy_cp, Skv * hierarchy_cp), # S_per_node, Different from dummy_da_config
                        'bs': bs,
                        'D': D,
                    }
                    da_config = Dist_Attn_Config.from_bsa_config(
                        dummy_da_config.bsa_config,     # full
                        shape_config=shape_config,
                        hierarchy=1,
                    )
                    profile_all_intra_full(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf, prof_db)
                    torch.distributed.barrier(gloo_global_group)

def step3_generate_inter_bsa_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db: Prof_DB, is_bypass_mode: bool = False):
    # Step3: Generate the inter-BSA; need all cpus on one node; (w cache/bypass)
    # inter_node_bsa_configs: List[Dict{CP: BSA_Config}]    # [DEPRECATED]
    # inter_node_bsa_configs: List[BSA_Config]
    
    # Prepare Inter_comp_profile_map(BSA) in m_config
    m_config = prof_db.m_config
    assert os.path.exists(prof_db.INTRA_BSA_EXE_PLANS_PROFILE), f'[ERROR]: INTRA_BSA_EXE_PLANS_PROFILE={prof_db.INTRA_BSA_EXE_PLANS_PROFILE} needs to exist'
    with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)
    m_config.update_inter_bsa_profile(intra_bsa_exe_plans_profile)
    
    inter_node_shape_configs = shape_config_dict['inter']
    hierarchy = 0   # (0, 1) -> (inter, intra)
    inter_plan_id = 0
    inter_exp_da_configs: List[dict] = []
    for exp_config in exp_configs:
        for inter_node_bsa_config in inter_node_bsa_configs:
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
                                inter_node_bsa_config, 
                                shape_config=shape_config,
                                hierarchy=hierarchy,
                            )
                            S_per_gpu = S_tot // da_config.tot_sp
                            if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                                continue
                            inter_exp_da_configs.append({
                                'exp_config': exp_config,
                                'da_config': da_config,
                            })
                            print_rank_0(f'inter_plan_id: {inter_plan_id}')
                            generate_inter_bsa_execution_plans(exp_config, da_config, prof_db, is_bypass_mode)
                            inter_plan_id += 1
    return inter_exp_da_configs

def step3_generate_inter_dense_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db: Prof_DB, is_bypass_mode: bool = False):
    # Step3: Generate the inter-BSA; need all cpus on one node; (w cache/bypass)
    # inter_node_bsa_configs: List[Dict{CP: BSA_Config}]    # [DEPRECATED]
    # inter_node_bsa_configs: List[BSA_Config]
    hierarchy = 0
    inter_node_full_configs, inter_node_causal_configs = split_dense_configs(inter_node_bsa_configs)
    
    # Prepare Inter_comp_profile_map(BSA) in m_config
    m_config = prof_db.m_config
    assert os.path.exists(prof_db.INTRA_BSA_EXE_PLANS_PROFILE), f'[ERROR]: INTRA_BSA_EXE_PLANS_PROFILE={prof_db.INTRA_BSA_EXE_PLANS_PROFILE} needs to exist'
    with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)
    m_config.update_inter_bsa_profile(intra_bsa_exe_plans_profile)
    
    # 1. For full, the same as step1 ✅
    dummy_shape_config = {
        'Nh': (1, 1),
        'S': (16 * 1024, 16 * 1024),    # total_S (smallest)
        'bs': 1,
        'D': 128,
    }
    intra_exp_da_full_configs: List[dict] = []
    for exp_config in exp_configs:  # fobs
        for intra_node_bsa_config in inter_node_full_configs:    # CPs
            if not (intra_node_bsa_config.bsa_repr.block_table_raw.shape == (1, 1) \
                and intra_node_bsa_config.bsa_repr.block_table_raw[0][0].value == Block_Type.FULL.value):
                assert False
            da_config = Dist_Attn_Config.from_bsa_config(
                intra_node_bsa_config, 
                shape_config=dummy_shape_config,
                hierarchy=hierarchy,
            )
            intra_exp_da_full_configs.append({
                'exp_config': exp_config,
                'da_config': da_config,
            })
            generate_general_full_execution_plans(exp_config, da_config, hierarchy, prof_db, is_bypass_mode)
    
    # 2. For causal, origin feature is implemented in `./scripts/schedule/search_engine_old.sh` with generate_inter_execution_plans ✅
    # [TODO]: support the cases where CP < split_q/kv
    inter_node_shape_configs = shape_config_dict['inter']
    hierarchy = 0   # (0, 1) -> (inter, intra)
    inter_plan_id = 0
    inter_exp_da_causal_configs: List[dict] = []
    for exp_config in exp_configs:
        for inter_node_causal_config in inter_node_causal_configs:
            assert inter_node_causal_config.bsa_repr.block_table_raw.shape == (1, 1) \
                and inter_node_causal_config.bsa_repr.block_table_raw[0, 0].value == Block_Type.CAUSAL.value
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
                                inter_node_causal_config, 
                                shape_config=shape_config,
                                hierarchy=hierarchy,
                            )
                            S_per_gpu = S_tot // da_config.tot_sp
                            if not (shape_config_dict['S_per_gpu_BOUND'][0] <= S_per_gpu <= shape_config_dict['S_per_gpu_BOUND'][1]):
                                continue
                            inter_exp_da_causal_configs.append({
                                'exp_config': exp_config,
                                'da_config': da_config,
                            })
                            print_rank_0(f'inter_plan_id: {inter_plan_id}')
                            generate_inter_bsa_execution_plans(exp_config, da_config, prof_db, is_bypass_mode)
                            inter_plan_id += 1
    return intra_exp_da_full_configs, inter_exp_da_causal_configs

def main(args):
    set_global_var('ARGS', args)
    # Initialize distribution
    ncclcomm_global, gloo_global_group = initialize_distribution()
    # Initialize Profile_DataBase
    prof_db = initialize_prof_db(gloo_global_group)
    
    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict, exp_configs = step0_top_down_decompose()
    if args.exp_class == 'dense_train': # full, causal
        # Step1: Generate execution plans for all dense at intra_SP=8; need all cpus on one node; (w cache/bypass) ✅
        intra_node_full_configs, intra_node_causal_configs = split_dense_configs(intra_node_bsa_configs)
        if torch.distributed.get_rank() == 0:
            step1_generate_intra_full_exe_plans(intra_node_full_configs, shape_config_dict, exp_configs, prof_db)       # Full
            step1_generate_intra_bsa_exe_plans(intra_node_causal_configs, shape_config_dict, exp_configs, prof_db)      # Causal
        torch.distributed.barrier(gloo_global_group)
        
        intra_exp_da_full_configs = step1_generate_intra_full_exe_plans(intra_node_full_configs, shape_config_dict, exp_configs, prof_db, is_bypass_mode=True)  # Full
        intra_exp_da_causal_configs = step1_generate_intra_bsa_exe_plans(intra_node_causal_configs, shape_config_dict, exp_configs, prof_db, is_bypass_mode=True)  # Bypass mode
        torch.distributed.barrier(gloo_global_group)
        # return      # Step1 End
        
        # Step2: Profile all dense at intra_SP=8; one node, one processor occupies one gpu and even cpus; (w cache/bypass) ✅
        if torch.cuda.is_available():
            step2_profile_intra_bsa_exe_plans(intra_exp_da_causal_configs, ncclcomm_global, gloo_global_group, prof_db) # For causal
            step2_profile_intra_full_exe_plans(intra_exp_da_full_configs, shape_config_dict, ncclcomm_global, gloo_global_group, prof_db) # For full
        # return  # Step2 End
        
        # Step3: Generate execution plans for all dense at inter_SP=2,4,8; need all cpus on one node; (w cache/bypass)
        if torch.distributed.get_rank() == 0:
            step3_generate_inter_dense_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db) # Full + Causal
    else:
        # Step1: Generate execution plans for all BSA at intra_SP=2,4,8; need all cpus on one node; (w cache/bypass)
        if torch.distributed.get_rank() == 0: #and not torch.cuda.is_available():
            step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict, exp_configs, prof_db)
        torch.distributed.barrier(gloo_global_group)
        # return    # Step1 End

        intra_exp_da_configs = step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict, exp_configs, prof_db, is_bypass_mode=True)  # Bypass mode
        torch.distributed.barrier(gloo_global_group)
        # return      # Step1 End

        # Step2: Profile all BSA at intra_SP=8; one node, one processor occupies one gpu and even cpus; (w cache/bypass)
        if torch.cuda.is_available():
            step2_profile_intra_bsa_exe_plans(intra_exp_da_configs, ncclcomm_global, gloo_global_group, prof_db)
        # return  # Step2 End
        
        # Step3: Generate execution plans for all BSA at inter_SP=2,4,8; need all cpus on one node; (w cache/bypass)
        if torch.distributed.get_rank() == 0:
            step3_generate_inter_bsa_exe_plans(inter_node_bsa_configs, shape_config_dict, exp_configs, prof_db)
    
if __name__ == '__main__':
    main(parse_args())