import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, Machine_Config, \
                                      get_profile_data, get_init_schedule_list, get_cc_optimal_schedule, get_cc_optimal_schedule_from_table
from search_algo.dependent_graph import Dependent_Graph
from search_algo.graph_transformation_engine import Graph_Transformation_Engine
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from search_algo.bsa_config import BSA_Repr, BSA_Config
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs
from search_algo.utils import combine_list_to_0, convert_block_table_to_value, parse_args, print_rank_0
from search_algo.database import Prof_DB
import torch
import torch.distributed as dist
from functools import partial
import socket
import tests
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from search_algo.workload_partition import solve_sparse_from_bsa
from search_algo.benchmark import benchmark_orchestrate_bsa
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func
import json
import random
from typing import List, Tuple, Union, Optional
from search_algo.exp_configs import step0_top_down_decompose
from search_algo.initialize import initialize_distribution, initialize_prof_db

# In-file global vars
DTYPE = torch.bfloat16
# End
def dummy_placeholder_op(*args, **kwargs):
    pass

# def get_intra_bsa_cc_optimal_schedule(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, m_config: Machine_Config) -> Dist_Attn_Config:
#     # [DEPRECATED]
#     fob = exp_config.fob
#     hierarchy = da_config.hierarchy   # (0, 1) -> (inter, intra)
#     assert hierarchy == 1, f"[ERROR]: (hierarchy={hierarchy}) should be 1 in 'get_intra_bsa_cc_optimal_schedule'"
#     DATABASE_ROOT = get_global_var('DATABASE_ROOT')
#     os.makedirs(DATABASE_ROOT, exist_ok=True)
#     INTRA_BSA_ALLOCATION_DB = f'{DATABASE_ROOT}/intra_bsa_allocation.json'
#     # os.makedirs(INTRA_BSA_ALLOCATION_DB, exist_ok=True)
#     if not os.path.exists(f'{INTRA_BSA_ALLOCATION_DB}'):
#         with open(f'{INTRA_BSA_ALLOCATION_DB}', 'w') as f:
#             json.dump({}, f)
#     key = f'fob={fob}_bsa_config={{{da_config.bsa_config}}}'  # [TODO]
#     print_rank_0(f'intra_bsa_allocation_key: {key}')
#     with open(f'{INTRA_BSA_ALLOCATION_DB}', 'r') as f:
#         intra_bsa_allocation_dict = json.load(f)
#     if key in intra_bsa_allocation_dict.keys():
#         print_rank_0(f'Bypassed !!!')
#         value = intra_bsa_allocation_dict[key]
#         schedule_table = np.array(value['schedule_table'], dtype=np.int32)
#         assert value['Par_D'] == schedule_table.shape[-1]
#         schedule_results = {
#             'CP': da_config.bsa_config.CP,
#             # 'cmap': da_config.bsa_config.cmap,
#             'table': schedule_table,
#         }
#         # print_rank_0(f'cmap: {da_config.bsa_config.cmap}') # None !!!
#     else:
#         print_rank_0(f'Not bypass !!!')
#         assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
#         schedule_results = solve_sparse_from_bsa(da_config.bsa_config, fob, hierarchy=hierarchy)
#         schedule_table = schedule_results['table']
#         # print_rank_0(f'schedule_table: {schedule_table.dtype}')    # int32
#         value = {
#             'Par_D': schedule_table.shape[-1],
#             'schedule_table': schedule_table.tolist(),
#         }
#         intra_bsa_allocation_dict[key] = value
#         with open(f'{INTRA_BSA_ALLOCATION_DB}', 'w') as f:
#             json.dump(intra_bsa_allocation_dict, f)
    
#     cc_optimal_schedule = get_cc_optimal_schedule_from_table(da_config, m_config, schedule_results)
    
#     if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
#         assert isinstance(cc_optimal_schedule, list)
#         cc_optimal_schedule = cc_optimal_schedule[0]
#     print_rank_0(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
#     return cc_optimal_schedule

def get_general_bsa_cc_optimal_schedule(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB) -> Dist_Attn_Config:
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
            
        schedule_results = solve_sparse_from_bsa(da_config.bsa_config, fob=fob, Par_D=Par_D, hierarchy=hierarchy)  # modify here !!!✅
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
    
def generate_intra_bsa_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB):
    exp_config.hierarchy = da_config.hierarchy = 1
    m_config = prof_db.m_config
    print_rank_0(f'da_config.shape_config: {da_config.shape_config}')
    # cc_optimal_schedule = get_intra_bsa_cc_optimal_schedule(exp_config, da_config, m_config)
    cc_optimal_schedule = get_general_bsa_cc_optimal_schedule(exp_config, da_config, prof_db)
    # exit(0)
    
    # Generate Intra_Execution_Plans:
    intra_bsa_exe_plans_dict_changed = False
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    
    #   1. Generate Dependent_Graph:
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
    
    #   2. Generate 4 types of Execution_Plans for ablations:
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # [TODO]: add Nhs and Ss to key_preffix !!!
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
            print_rank_0(f'Not bypass !!!')
            # assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type)
            execute_plan.print_lp_result()
            # Dump Execution_Plan:
            plan_id = max(intra_bsa_exe_plans_dict.values()) + 1 if intra_bsa_exe_plans_dict else 0
            intra_bsa_exe_plans_dict[key] = plan_id
            intra_bsa_exe_plans_dict_changed = True
            plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        else:
            # print_rank_0(f'Bypassed !!!')
            plan_id = intra_bsa_exe_plans_dict[key]
            plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'rb') as fin:
                execute_plan: Execution_Plan = pickle.load(fin)
            # execute_plan.print_lp_result()
            print_rank_0(f'end_time={execute_plan.get_end_time():.3e}')
        
        # w Kernel Tile Execution_Plans:
        KERNEL_TILE_TYPE = 'w_kernel_tile'
        print_rank_0(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:')
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print_rank_0(f'intra_bsa_exe_plan_key: {key}')
        if key not in intra_bsa_exe_plans_dict.keys():
            print_rank_0(f'Not bypass !!!')
            # assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
            gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
            execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=plan_type)
            if execute_plan is None:    # No feasible transformations
                assert False
                continue
            assert isinstance(execute_plan, Execution_Plan)
            
            plan_name = f'{execute_plan.get_plan_name()}_fused'
            if plan_type == 'ablation1':
                plan_name = f'{plan_name}_{plan_type}'
            # Dump Execution_Plan:
            plan_id = max(intra_bsa_exe_plans_dict.values()) + 1 if intra_bsa_exe_plans_dict else 0
            intra_bsa_exe_plans_dict[key] = plan_id
            intra_bsa_exe_plans_dict_changed = True
            plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        else:
            # print_rank_0(f'Bypassed !!!')
            plan_id = intra_bsa_exe_plans_dict[key]
            plan_file = f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'rb') as fin:
                execute_plan: Execution_Plan = pickle.load(fin)
            # execute_plan.print_lp_result()
            print_rank_0(f'end_time={execute_plan.get_end_time():.3e}')
    
    if intra_bsa_exe_plans_dict_changed:
        # assert not torch.cuda.is_available(), f'intra_bsa_exe_plans_dict should not be changed in GPU nodes'
        with open(f'{prof_db.INTRA_BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump(intra_bsa_exe_plans_dict, f)

def profile_all_intra_BSA(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor, prof_db: Prof_DB):
    PROC_INFO = get_global_var(f'PROC_INFO')
    # [TODO]: Support baseline here !!! @yqg
    # baseline_funcs = [
    #     ring_flash_attn_func,
    #     zigzag_ring_flash_attn_func,      # baseline
    # ]
    
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_num = PROC_INFO['node_num']
    node_id = PROC_INFO['nodeid']
    rank = PROC_INFO['rank']
    
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'
    
    # experiment variables
    WARMUP, NUM_ITER = 11, 20 # most, best performance for most cases
    WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    
    # Prepare database
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)

    # Generate inter_comp_plans_dicts
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    key_suffixes = [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                        for KERNEL_SCHEDULE_TYPE in ['ILP', 'Flexflow'] \
                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']]
    keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
    # End
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    for key in keys:
        # Create dummy inter_bsa_execution_plan
        inter_bsa_execution_plan: Optional[Execution_Plan] = Execution_Plan.create_one_node_exe_plan(exp_config.fob)
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        # load exe_plan
        plan_id = intra_bsa_exe_plans_dict[key]
        with open(f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
            intra_bsa_execution_plan: Execution_Plan = pickle.load(fin)
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
        
    is_bypass = all([key in intra_bsa_exe_plans_profile.keys() for key in keys])
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
            warmup=WARMUP, num_iter=NUM_ITER, log=True, mode='profile',
        )
        bench_results = benchmark_op(use_cudagraph=False)   # List[[Tflops/s, s]] for rank0, List[[None, None]] for others
        assert len(bench_results) == len(inter_comp_plans_dicts)
        # Save bench_results to profile json file
        if torch.distributed.get_rank() == 0:
            for key, bench_result in zip(keys, bench_results):
                assert key not in intra_bsa_exe_plans_profile, f'profile_all_intra_BSA is profiled by grained of all ablation tests !!!'
                intra_bsa_exe_plans_profile[key] = bench_result
            # print_rank_0(f'intra_bsa_exe_plans_profile: {intra_bsa_exe_plans_profile}')
            with open(prof_db.INTRA_BSA_EXE_PLANS_PROFILE, 'w') as f:
                json.dump(intra_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')
    
    
def generate_inter_bsa_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB):
    DATABASE_ROOT = get_global_var('DATABASE_ROOT')
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
    exp_config.hierarchy = da_config.hierarchy = 0  # Inter
    
    # Prepare Inter_comp_profile_map(BSA) in m_config
    m_config: Machine_Config = get_profile_data(da_config.SP, exp_config.hierarchy) # Comm is useful but comp is useless here !!! (in database)
    INTRA_BSA_EXE_PLANS_PROFILE = f'{DATABASE_ROOT}/intra_bsa_exe_plans_profile.json'
    assert os.path.exists(INTRA_BSA_EXE_PLANS_PROFILE), f'[ERROR]: INTRA_BSA_EXE_PLANS_PROFILE={INTRA_BSA_EXE_PLANS_PROFILE} needs to exist'
    with open(f'{INTRA_BSA_EXE_PLANS_PROFILE}', 'r') as f:
        intra_bsa_exe_plans_profile = json.load(f)
    # m_config.update_inter_bsa_profile(intra_bsa_exe_plans_profile)
    # prof_db.update_m_config(m_config)
    print_rank_0(f'da_config.shape_config Inter: {da_config.shape_config}')
    
    # Calc optimal schedule
    cc_optimal_schedule = get_general_bsa_cc_optimal_schedule(exp_config, da_config, m_config)
    # exit(0)
    # Generate Inter_Execution_Plans:
    INTER_BSA_EXE_PLANS_DIR = f'{DATABASE_ROOT}/{CLUSTER_NAME}/{PLATFORM}/inter_bsa_exe_plans'
    INTER_BSA_EXE_PLANS_KV = f'{DATABASE_ROOT}/inter_bsa_exe_plans_kv.json'
    os.makedirs(INTER_BSA_EXE_PLANS_DIR, exist_ok=True)
    if not os.path.exists(f'{INTER_BSA_EXE_PLANS_KV}'):
        with open(f'{INTER_BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump({}, f)
    inter_bsa_exe_plans_dict_changed = False
    with open(f'{INTER_BSA_EXE_PLANS_KV}', 'r') as f:
        inter_bsa_exe_plans_dict = json.load(f)
    
    #   1. Generate Dependent_Graph:
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob, is_inter_bsa=True, bsa_comp_key_suffix=f'_ablation=(w_kernel_tile,ILP)')
    # [TODO]: different d_graph regarding to ablations !!! But we use ('ILP', 'w kernel tile') currently.
    
    #   2. Generate 2 types of Execution_Plans for ablations (no 'w kernel tile')
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    plan_types = ['ILP', 'Flexflow']
    for plan_type in plan_types:
        # KERNEL_SCHEDULE_TYPE = "ILP" if plan_type == "automatic" else "Flexflow"
        KERNEL_SCHEDULE_TYPE = plan_type
        # w/o Kernel Tile Execution_Plan:
        KERNEL_TILE_TYPE = 'w/o_kernel_tile'
        print_rank_0(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:')
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print_rank_0(f'inter_bsa_exe_plan_key: {key}')
        if key not in inter_bsa_exe_plans_dict.keys():
            print_rank_0(f'Not bypass !!!')
            # assert not torch.cuda.is_available(), f'All GPU workloads should be bypassed in GPU nodes'
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type) # [TODO]: m_config is error !!!
            execute_plan.print_lp_result()
            # Dump Execution_Plan:
            plan_id = max(inter_bsa_exe_plans_dict.values()) + 1 if inter_bsa_exe_plans_dict else 0
            inter_bsa_exe_plans_dict[key] = plan_id
            inter_bsa_exe_plans_dict_changed = True
            plan_file = f'{INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        else:
            print_rank_0(f'Bypassed !!!')
        
    
    if inter_bsa_exe_plans_dict_changed:
        # assert not torch.cuda.is_available(), f'inter_bsa_exe_plans_dict should not be changed in GPU nodes'
        with open(f'{INTER_BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump(inter_bsa_exe_plans_dict, f)

def step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict: dict, exp_configs, prof_db):
    # Step1: Generate the intra-BSA; need all cpus on one node; (w cache/bypass)
    intra_node_shape_configs = shape_config_dict['intra']
    intra_plan_id = 0
    intra_da_configs: List[Dist_Attn_Config] = []
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
                            intra_da_configs.append(da_config)
                            print_rank_0(f'intra_plan_id: {intra_plan_id}')
                            generate_intra_bsa_execution_plans(exp_config, da_config, prof_db)
                            intra_plan_id += 1
    return intra_da_configs

def step2_profile_intra_bsa_exe_plans(intra_da_configs, exp_configs, ncclcomm_global, gloo_global_group, prof_db):
    MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    for da_config in intra_da_configs:
        MAX_S_perG = max(MAX_S_perG, max(da_config.S_per_gpu))
        MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
        MAX_D = max(MAX_D, da_config.shape_config['D'])
        MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
    
    tensor_buf = torch.empty(
        (MAX_bs * MAX_S_perG * MAX_NH * MAX_D * 4) * 3                       # k, v, dk, dv
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    + (MAX_bs * MAX_S_perG * MAX_NH * (MAX_D * 2) + (1 * (2 + 1))) * 2,    # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    args = parse_args()
    for exp_config in exp_configs:  # fobs
        for da_config in intra_da_configs:
            profile_all_intra_BSA(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf, prof_db)

def step3_generate_inter_bsa_exe_plans(inter_node_bsa_configs, intra_node_shape_configs, exp_configs, prof_db):
    # Step3: Generate the inter-BSA; need all cpus on one node; (w cache/bypass)
    # inter_node_bsa_configs: List[Dict{CP: BSA_Config}]    # [DEPRECATED]
    # inter_node_bsa_configs: List[BSA_Config]
    hierarchy = 1   # (0, 1) -> (inter, intra)
    inter_plan_id = 0
    inter_da_configs: List[Dist_Attn_Config] = []
    for exp_config in exp_configs:
        for inter_node_bsa_config in inter_node_bsa_configs:
            for Nh in intra_node_shape_configs['Nhs']:
                for S_per_node in intra_node_shape_configs['Ss']:
                    for bs in intra_node_shape_configs['BSs']:
                        for D in intra_node_shape_configs['Ds']:
                            inter_node_S = S_per_node * inter_node_bsa_config.CP[1]
                            # print_rank_0(f'S_per_node: {S_per_node}; inter_node_S: {inter_node_S}; inter_node_bsa_config.CP: {inter_node_bsa_config.CP}')
                            shape_config = {
                                'Nh': (Nh, Nh),
                                'S': (inter_node_S, inter_node_S),  # inter_node_S
                                'bs': bs,
                                'D': D,
                            }
                            da_config = Dist_Attn_Config.from_bsa_config(
                                inter_node_bsa_config, 
                                shape_config=shape_config,
                                hierarchy=hierarchy,
                            )
                            inter_da_configs.append(da_config)
                            print_rank_0(f'inter_plan_id: {inter_plan_id}')
                            generate_inter_bsa_execution_plans(exp_config, da_config, prof_db)
                            inter_plan_id += 1
    return inter_da_configs

def main():
    # Initialize distribution
    ncclcomm_global, gloo_global_group = initialize_distribution()
    # Initialize Profile_DataBase
    prof_db = initialize_prof_db()
    
    # if torch.distributed.get_rank() == 0:
    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict, exp_configs = step0_top_down_decompose()
    if torch.distributed.get_rank() == 0: #and not torch.cuda.is_available():
        intra_da_configs = step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict, exp_configs, prof_db)
    torch.distributed.barrier(gloo_global_group)
    return    # Step1 End

    intra_da_configs = step1_generate_intra_bsa_exe_plans(intra_node_bsa_configs, shape_config_dict, exp_configs, prof_db)  # Bypass mode
    torch.distributed.barrier(gloo_global_group)

    # Step2: Profile all BSA at intra_SP=8; one node, one processor occupies one gpu and even cpus; (w cache/bypass)
    if torch.cuda.is_available():
        step2_profile_intra_bsa_exe_plans(intra_da_configs, exp_configs, ncclcomm_global, gloo_global_group, prof_db)
    return  # Step2 End
    
    # Step3: Generate execution plans for all BSA at inter_SP=2,4,8; need all cpus on one node; (w cache/bypass)  [TODO]
    if torch.distributed.get_rank() == 0:
        step3_generate_inter_bsa_exe_plans(inter_node_bsa_configs, intra_node_shape_configs, exp_configs, prof_db)    
    
if __name__ == '__main__':
    main()