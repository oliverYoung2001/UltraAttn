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
from search_algo.benchmark import benchmark_orchestrate_bsa, prepare_inter_comp_plans
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func
import json
import random
from typing import List, Tuple, Union, Optional
from search_algo.exp_configs import step0_top_down_decompose
from search_algo.initialize import initialize_distribution
from orchestrated_attn.global_vars import set_global_var as set_global_var_orch
from orchestrated_attn.global_vars import get_global_var as get_global_var_orch

# In-file global vars
DTYPE = torch.bfloat16
# End
def dummy_placeholder_op(*args, **kwargs):
    pass

def profile_all_inter_BSA(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor):
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
    # print(f'[RANK{torch.distributed.get_rank()}] Before gloo_global_group !!!', flush=True)
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
    
    inter_comp_profile_map = None
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
    
    # Generate inter_comp_plans_dicts
    inter_comp_plans_dicts = []
    # Prepare database
    DATABASE_ROOT = get_global_var('DATABASE_ROOT')
    INTER_BSA_EXE_PLANS_DIR = f'{DATABASE_ROOT}/{CLUSTER_NAME}/{PLATFORM}/inter_bsa_exe_plans'
    INTER_BSA_EXE_PLANS_KV = f'{DATABASE_ROOT}/inter_bsa_exe_plans_kv.json'
    with open(f'{INTER_BSA_EXE_PLANS_KV}', 'r') as f:
        inter_bsa_exe_plans_dict = json.load(f)
    
    INTER_BSA_EXE_PLANS_PROFILE = f'{DATABASE_ROOT}/inter_bsa_exe_plans_profile.json'
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(f'{INTER_BSA_EXE_PLANS_PROFILE}'):
            with open(f'{INTER_BSA_EXE_PLANS_PROFILE}', 'w') as f:
                json.dump({}, f)
    torch.distributed.barrier(gloo_global_group)
    with open(f'{INTER_BSA_EXE_PLANS_PROFILE}', 'r') as f:
        inter_bsa_exe_plans_profile = json.load(f)
    # End

    key_suffixes = [f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})' \
                        for KERNEL_SCHEDULE_TYPE in ['ILP', 'Flexflow'] \
                            for KERNEL_TILE_TYPE in ['w/o_kernel_tile']]
    keys = [f'{key_preffix}{key_suffix}' for key_suffix in key_suffixes]
    inter_bsa_execution_plans = []  # inter_bsa_execution_plans and inter_comp_plans_dicts are bijective !!!
    inter_comp_plans_dicts = []
    for key in keys:
        # load exe_plan
        plan_id = inter_bsa_exe_plans_dict[key]
        with open(f'{INTER_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
            inter_bsa_execution_plan: Execution_Plan = pickle.load(fin)
        inter_bsa_execution_plans.append(inter_bsa_execution_plan)
        # inter_comp_plans_dict: {intra_bsa_key: intra_bsa_exe_plan}
        # intra_bsa_key = ((relative_Sq, relative_Skv), str(da_config.bsa_config.bsa_repr))
        inter_comp_plans_dict = prepare_inter_comp_plans(inter_bsa_execution_plan)        
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
    is_bypass = all([key in inter_bsa_exe_plans_profile.keys() for key in keys])
    if not is_bypass:
        print_rank_0(f'Not bypass !!!')
        # Execution:
        # 1 baselines
        # [TODO]
        
        # 2 orchestrated_attn_func:
        # [TODO]: check corretness of da_config✅&exp_configs✅
        # [TODO]: run through this
        benchmark_op = partial(benchmark_orchestrate_bsa,
            args, orchestrated_attn_func, da_config, exp_config, tensor_buf, 
            inter_bsa_execution_plans=inter_bsa_execution_plans, inter_comp_plans_dicts=inter_comp_plans_dicts,
            global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
            warmup=WARMUP, num_iter=NUM_ITER, log=True, mode='test', 
        )
        bench_results = benchmark_op(use_cudagraph=False)   # List[[Tflops/s, s]] for rank0, List[[None, None]] for others
        assert len(bench_results) == len(inter_comp_plans_dicts)
        # Save bench_results to profile json file
        if torch.distributed.get_rank() == 0:
            for key, bench_result in zip(keys, bench_results):
                assert key not in inter_bsa_exe_plans_profile, f'profile_all_inter_BSA is profiled by grained of all ablation tests !!!'
                inter_bsa_exe_plans_profile[key] = bench_result
            with open(f'{INTER_BSA_EXE_PLANS_PROFILE}', 'w') as f:
                json.dump(inter_bsa_exe_plans_profile, f) 
    else:
        print_rank_0(f'Bypassed !!!')
    

def step4_profile_inter_bsa_exe_plans(inter_node_bsa_configs, intra_node_shape_configs, exp_configs, ncclcomm_global, gloo_global_group):
    # inter_node_bsa_configs: List[Dict{CP: BSA_Config}]    # [DEPRECATED]
    # inter_node_bsa_configs: List[BSA_Config]
    hierarchy = 0   # (0, 1) -> (inter, intra)
    inter_da_configs: List[Dist_Attn_Config] = []
    for exp_config in exp_configs:
        for inter_node_bsa_config in inter_node_bsa_configs:
            for Nh in intra_node_shape_configs['Nhs']:
                for S_per_node in intra_node_shape_configs['Ss']:
                    for bs in intra_node_shape_configs['BSs']:
                        for D in intra_node_shape_configs['Ds']:
                            inter_node_S = S_per_node * inter_node_bsa_config.CP[1]
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
    
    MAX_S_perG, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
    for da_config in inter_da_configs:
        MAX_S_perG = max(MAX_S_perG, max(da_config.S_per_gpu))
        MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
        MAX_D = max(MAX_D, da_config.shape_config['D'])
        MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
    
    tensor_buf = torch.empty(
    #     (MAX_bs * MAX_S * MAX_NH * MAX_D * 4) * 3                       # k, v, dk, dv
    # + (MAX_bs * MAX_S * MAX_NH * (MAX_D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    # + (MAX_bs * MAX_S * MAX_NH * (MAX_D * 2) + (1 * (2 + 1))) * 2,    # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
        ((MAX_bs * MAX_S_perG * MAX_NH * (MAX_D + 1)) * (4 * 8) * (4)),
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    print_rank_0(f'MAX_S_perG={MAX_S_perG}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
    print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    args = parse_args()
    inter_plan_id = 0
    
    # gloo_global_group = dist.new_group(ranks=list(range(PROC_INFO['world_size'])), backend='gloo')
    for exp_config in exp_configs:  # fobs
        for da_config in inter_da_configs:
            print_rank_0(f'[inter_plan_id {inter_plan_id}]')
            profile_all_inter_BSA(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf)
            torch.distributed.barrier(gloo_global_group)
            inter_plan_id += 1

def main():
    # Initialize distribution
    ncclcomm_global, gloo_global_group = initialize_distribution()
    
    # Initialize Profile_DataBase
    prof_db = Prof_DB() # [TODO]: finish profile database

    # if torch.distributed.get_rank() == 0:
    inter_node_bsa_configs, intra_node_bsa_configs, intra_node_shape_configs, exp_configs = step0_top_down_decompose()
    
    # Step4: Profile all BSA at inter_SP; multiple nodes, one processor occupies one gpu and even cpus; (w cache/bypass)
    # [TODO]
    step4_profile_inter_bsa_exe_plans(inter_node_bsa_configs, intra_node_shape_configs, exp_configs, ncclcomm_global, gloo_global_group)
    
if __name__ == '__main__':
    main()