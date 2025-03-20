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

# In-file global vars
DTYPE = torch.bfloat16
placeholder_op = None
# End
def dummy_placeholder_op(*args, **kwargs):
    pass

def parse_slurm_tasks_per_node(tasks_per_node):
    # 4(x2), 8, ...
    return int(tasks_per_node.split('(')[0])

def get_proc_info():
    if os.getenv('SLURM_PROCID', None) is not None:    # launch with Slurm
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        ip = os.environ['SLURM_STEP_NODELIST']
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.environ['SLURM_CLUSTER_NAME']
        nodeid = int(os.environ['SLURM_NODEID'])
        nodename = os.environ['SLURMD_NODENAME']
        tasks_per_node = parse_slurm_tasks_per_node(os.environ['SLURM_TASKS_PER_NODE'])
        
    elif os.getenv('OMPI_COMM_WORLD_RANK', None) is not None: # launch with OpenMPI
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # ip = os.environ['SLURM_STEP_NODELIST']
        ip = None
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        # nodeid = int(os.environ['SLURM_NODEID'])
        # nodename = os.environ['SLURMD_NODENAME']
        nodename = None
        # tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
        tasks_per_node = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        nodeid = rank // tasks_per_node
        
    else:   # [NOTE]: assume that execute from python
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        hostname = socket.gethostname()
        nodename = None
        nodeid = 0
        world_size = 1
        tasks_per_node = 1
        rank = 0
        local_rank = 0
        hostip = socket.gethostbyname(hostname)
        ip = None
        print(f'hostip: {hostip}')
        os.environ['MASTER_ADDR'] = hostip
        os.environ['MASTER_PORT'] = str(random.randint(0, 12000) + 10000)
        os.environ['CLUSTER_NAME'] = 'fit'
        os.environ['PLATFORM'] = 'A800'
        # raise Exception("Unknown Launcher !!!")
    proc_info = {
        'clustername': clustername,
        'hostname': hostname,
        'nodename': nodename,
        'nodeid': nodeid,
        'world_size': world_size,
        'tasks_per_node': tasks_per_node,
        'rank': rank,
        'local_rank': local_rank,
        'hostip': hostip,
        'ip': ip,
        'deviceid': local_rank,
    }
    proc_info['node_num'] = world_size // tasks_per_node
    # print(f'proc_info: {proc_info}')
    return proc_info

def initialize_distribution():
    PROC_INFO = get_proc_info()
    set_global_var(f'PROC_INFO', PROC_INFO)
    # print(f'PROC_INFO: {PROC_INFO}')
    
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    # MASTER_ADDR = 'localhost'
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend=BACKEND,
        # init_method=init_method, 
        rank=PROC_INFO['rank'], 
        world_size=PROC_INFO['world_size'])
    gloo_global_group = dist.new_group(ranks=list(range(PROC_INFO['world_size'])), backend='gloo')
    ncclcomm_global = PyNcclCommunicator(gloo_global_group, ranks=list(range(PROC_INFO['world_size'])), device=PROC_INFO['local_rank'])
    # [NOTE]: we create a gloo global group because we use it to barrier in benchmark_orchestrate to prevent cudagraph overlapped with nccl ops !!!
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f'rank{rank}, world_size{world_size}, hostname: {socket.gethostname()}')
    # initialize_distributed()    # used by lightseq

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    
    # preprocess placeholder_op
    global placeholder_op
    SYNC_SIZE = 8 * pow(1024, 3) # 8GB
    sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=device)
    placeholder_op = partial(ncclcomm_global.all_reduce, sync_tensor)
    return ncclcomm_global, gloo_global_group

def get_exp_configs():
    # plan_type = 'ILP'         # automatic
    plan_type = 'manual'        # for noncausal !!!
    # plan_type = 'Flexflow'    # ablation1
    MAX_QUEUE_SIZE = 100
    fobs = [
        0,
        # 1,
    ]
    # hierarchy = 0  # 0: intra-machine, 1: inter-machine
    hierarchy = None    # define in exps !!!
    transform_mode = 'bf'       # Enumerate all possible transformations
    transform_mode = 'greedy'   # Apply transformations greedily
    exp_configs = []
    for fob in fobs:
        exp_configs.append(Evaluation_Configs(plan_type, MAX_QUEUE_SIZE, fob, hierarchy=hierarchy, transform_mode=transform_mode))
    return exp_configs

def get_configs():
    # da_config: SP=(1,8),Sg=(1024,1024),S=(8192,8192),Nh=(1,1),bs=1,D=128,causal=True,hierarchy=None

    hierarchy = None    # define in exps !!!
    MAX_SP = 8 * 8
    # for Intra:
    # SP0, SP1 = 1, 1
    # SP0, SP1 = 1, 2
    # SP0, SP1 = 1, 4
    # SP0, SP1 = 1, 8
    # Sq = Skv = 16 * 1024   # 16k
    # Sq = Skv = 8 * 1024   # 8k
    
    
    # # for Inter:
    # SP0, SP1 = 1, 8
    # SP0, SP1 = 2, 8
    # # # SP0, SP1 = 3, 8
    # SP0, SP1 = 4, 8 # 512K (global) -> 16K (S per gpu)
    # # # # SP0, SP1 = 6, 8
    # SP0, SP1 = 8, 8 # 512K (global) -> 8K (S per gpu)
    # # SP0, SP1 = 16, 8
    PLATFORM = os.getenv(f'PLATFORM')
    if PLATFORM == 'A800':
        CPs = [
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
        ]
    elif PLATFORM == 'H800':
        CPs = [
            (8, 1),
            (8, 2),
            (8, 4),
        ]
    else:
        raise Exception(f"Unknown PLATFORM={PLATFORM}")

    Ss_per_gpu = [
        # 256, 
        # 512, 
        # 1 * 1024, 
        # 2 * 1024, 
        # 4 * 1024, 
        # 8 * 1024, 
        # 16 * 1024, 
        # 32 * 1024,    # fused failed on 8 * 8 !!!
        64 * 1024,    # fused failed on 4 * 8 !!!
    ]    # S per GPU -> S total
    Ss = [
        64 * 1024,
    ]    # S total
    # Ss = [16 * 1024]    # S per GPU

    # Nhq = Ng = 32
    # # Nhq = Ng = 1
    Nhs = [
        1,
        32,
    ]
    bs = 1
    D = 128
    # causal = False
    # causal = True
    use_BSA = False
    use_BSA = True  # prior than 'causal', this will hide 'causal'
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks(l&r), global_blocks(l&r), (replicate)
    # 8_8_lg_1-8_0_0&1
    # 8_8_lg_1-4_3_0_2
    # BSA_patterns = [    # a dict
    #     # # stride_16_4_3 (8x2)
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2},
    #     # stride_16_4_3 (8x4)
    #     {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 1},
    #     # # stride_16_4_3 (8x8)
    #     # # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 2), 'global_blocks': (0, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 1), 'global_blocks': (0, 0), 'replicate': 1},
    #     # # lg_16_1_1 (8x2)
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
    #     # # lg_16_1_1 (8x4)
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
    #     # # lg_16_1_1 (8x8)
    #     # # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
    #     # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
    # ]
    cmap_finest = np.arange(MAX_SP)   # max CP !!!
    cmap_finest = None
    global_bsa_reprs = []
    # 1. BSA_Repr for stride(1/16, 4, 3) (after remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(0), cmap=cmap_finest))
    # 2. BSA_Repr for local+global(1/16, 1, 1) (no remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(1), cmap=cmap_finest))
    # [TODO]: Create global_bsa_configs from global_bsa_reprs   ✅
    # [TODO]: break global_bsa_reprs down to node-level bsa_repr&bsa_configs !!! ✅
    global_bsa_configs = []
    # intra_node_bsa_configs = set()
    intra_node_bsa_configs = []
    for i, gbr in enumerate(global_bsa_reprs):
        global_bsa_configs.append({})
        for CP in CPs:
            # print(f'CP: {CP}', flush=True)
            # print(f'gbr.minimum_Par_D: {gbr.minimum_Par_D}', flush=True)    # 16
            global_bsa_config = BSA_Config(
                None, None,
                {
                    'bsa_repr': gbr,
                    'CP': CP,
                }
            )
            global_bsa_configs[-1][CP] = global_bsa_config
            node_bsa_configs = split_to_node_configs(global_bsa_config)
            # intra_node_bsa_configs |= node_bsa_configs
            combine_list_to_0(intra_node_bsa_configs, node_bsa_configs)
    
    # filter out all empty configs
    filtered_configs = []
    for c in intra_node_bsa_configs:
        if not c.bsa_repr.check_empty():
            filtered_configs.append(c)
    intra_node_bsa_configs = filtered_configs
    for intra_node_bsa_config in intra_node_bsa_configs:
        block_table_value = convert_block_table_to_value(intra_node_bsa_config.block_table)
        print(f'CP: {intra_node_bsa_config.CP}, minimum_Par_D: {intra_node_bsa_config.bsa_repr.minimum_Par_D}\n{block_table_value}', flush=True)
    shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss,
        'BSs': [bs],
        'Ds': [D],
    }
    return global_bsa_configs, intra_node_bsa_configs, shape_configs

def get_intra_bsa_cc_optimal_schedule(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, m_config: Machine_Config) -> Dist_Attn_Config:
    DATABASE_ROOT = get_global_var('DATABASE_ROOT')
    os.makedirs(DATABASE_ROOT, exist_ok=True)
    INTRA_BSA_ALLOCATION_DB = f'{DATABASE_ROOT}/intra_bsa_allocation.json'
    # os.makedirs(INTRA_BSA_ALLOCATION_DB, exist_ok=True)
    if not os.path.exists(f'{INTRA_BSA_ALLOCATION_DB}'):
        with open(f'{INTRA_BSA_ALLOCATION_DB}', 'w') as f:
            json.dump({}, f)
    key = f'fob={exp_config.fob}_bsa_config={{{da_config.bsa_config}}}'  # [TODO]
    print(f'intra_bsa_allocation_key: {key}', flush=True)
    with open(f'{INTRA_BSA_ALLOCATION_DB}', 'r') as f:
        intra_bsa_allocation_dict = json.load(f)
    if key in intra_bsa_allocation_dict.keys():
        print(f'Bypassed !!!', flush=True)
        value = intra_bsa_allocation_dict[key]
        schedule_table = np.array(value['schedule_table'], dtype=np.int32)
        assert value['Par_D'] == schedule_table.shape[-1]
        schedule_results = {
            'CP': da_config.bsa_config.CP,
            # 'cmap': da_config.bsa_config.cmap,
            'table': schedule_table,
        }
        # print(f'cmap: {da_config.bsa_config.cmap}', flush=True) # None !!!
    else:
        print(f'Not bypass !!!', flush=True)
        schedule_results = solve_sparse_from_bsa(da_config.bsa_config)
        schedule_table = schedule_results['table']
        # print(f'schedule_table: {schedule_table.dtype}', flush=True)    # int32
        value = {
            'Par_D': schedule_table.shape[-1],
            'schedule_table': schedule_table.tolist(),
        }
        intra_bsa_allocation_dict[key] = value
        with open(f'{INTRA_BSA_ALLOCATION_DB}', 'w') as f:
            json.dump(intra_bsa_allocation_dict, f)
    
    cc_optimal_schedule = get_cc_optimal_schedule_from_table(da_config, m_config, schedule_results)
    
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
    return cc_optimal_schedule
    
def generate_intra_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB):
    exp_config.hierarchy = da_config.hierarchy = 1
    m_config = get_profile_data(da_config.SP, exp_config.hierarchy)
    prof_db.update_m_config(m_config)
    print(f'da_config.shape_config: {da_config.shape_config}', flush=True)
    cc_optimal_schedule = get_intra_bsa_cc_optimal_schedule(exp_config, da_config, m_config)
    # exit(0)
    
    # Generate Intra_Execution_Plans:
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)

    DATABASE_ROOT = get_global_var('DATABASE_ROOT')
    INTRA_BSA_EXE_PLANS_DIR = f'{DATABASE_ROOT}/{CLUSTER_NAME}/{PLATFORM}/intra_bsa_exe_plans'
    INTRA_BSA_EXE_PLANS_KV = f'{DATABASE_ROOT}/intra_bsa_exe_plans_kv.json'
    os.makedirs(INTRA_BSA_EXE_PLANS_DIR, exist_ok=True)
    if not os.path.exists(f'{INTRA_BSA_EXE_PLANS_KV}'):
        with open(f'{INTRA_BSA_EXE_PLANS_KV}', 'w') as f:
            json.dump({}, f)
    with open(f'{INTRA_BSA_EXE_PLANS_KV}', 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    
    #   1. Generate Dependent_Graph:
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
    
    #   2. Generate 4 types of Execution_Plans for ablations:
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'
    # [TODO]: add Nhs and Ss to key_preffix !!!
    # plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    plan_types = ['ILP', 'Flexflow']
    for plan_type in plan_types:
        # KERNEL_SCHEDULE_TYPE = "ILP" if plan_type == "automatic" else "Flexflow"
        KERNEL_SCHEDULE_TYPE = plan_type
        # w/o Kernel Tile Execution_Plan:
        KERNEL_TILE_TYPE = 'w/o_kernel_tile'
        print(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:', flush=True)
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print(f'intra_bsa_exe_plan_key: {key}', flush=True)
        if key not in intra_bsa_exe_plans_dict.keys():
            print(f'Not bypass !!!', flush=True)
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type)
            execute_plan.print_lp_result()
            # Dump Execution_Plan:
            plan_id = max(intra_bsa_exe_plans_dict.values()) + 1 if intra_bsa_exe_plans_dict else 0
            intra_bsa_exe_plans_dict[key] = plan_id
            plan_file = f'{INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        else:
            print(f'Bypassed !!!', flush=True)
        
        # w Kernel Tile Execution_Plans:
        KERNEL_TILE_TYPE = 'w_kernel_tile'
        print(f'{KERNEL_TILE_TYPE}, {KERNEL_SCHEDULE_TYPE}:', flush=True)
        key_suffix = f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})'
        key = f'{key_preffix}{key_suffix}'
        print(f'intra_bsa_exe_plan_key: {key}', flush=True)
        if key not in intra_bsa_exe_plans_dict.keys():
            print(f'Not bypass !!!', flush=True)
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
            plan_file = f'{INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        else:
            print(f'Bypassed !!!', flush=True)
            
    with open(f'{INTRA_BSA_EXE_PLANS_KV}', 'w') as f:
        json.dump(intra_bsa_exe_plans_dict, f)

def profile_all_intra_BSA(args, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, ncclcomm_global, gloo_global_group, \
                          tensor_buf: torch.Tensor):
    PROC_INFO = get_global_var(f'PROC_INFO')
    # [TODO]: Support baseline here !!! @yqg
    # baseline_funcs = [
    #     ring_flash_attn_func,
    #     zigzag_ring_flash_attn_func,      # baseline
        
    #     # zigzag_ring_flash_attn_func_opt,  # sol1
    #     stripe_flash_attn_func,
        
    #     # lightseq_attn_func,
    #     # flash_attn_func,
    #     # hierarchy_attn_func,                # one case
    #     # overlapped_hierarchy_attn_func,     # another case
    # ]
    
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_num = PROC_INFO['node_num']
    rank = PROC_INFO['rank']
    
    assert local_size == 8, f'[ERROR]: Now not support for local_size({local_size}) intra-node not equal to 8'
    
    
    # # BSA_configs = [BSA_Config.from_dict(**p) for p in BSA_patterns_dict]
    # BSA_configs = [BSA_Config.from_dict(p) for p in BSA_patterns_dict]
    
    # experiment variables
    WARMUP, NUM_ITER = 11, 20 # most, best performance for most cases
    WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    # WARMUP, NUM_ITER = 0, 1 # [DEBUG]
    
    # S_BOUND = [256, 64 * 1024]  # lower-bound and upper-bound of S per GPU, for (1, 8)
    # # S_BOUND = [256, 16 * 1024] # for debug
    # # S_BOUND = [16 * 1024, 16 * 1024] # for debug
    # S_BOUND = [64 * 1024, 64 * 1024]

    inter_comp_profile_map = None
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
    
    # Generate inter_comp_plans_dicts
    inter_comp_plans_dicts = []
    key_preffix = f'fob={exp_config.fob}_CP={da_config.bsa_config.CP}_shape_config={{{da_config.get_shape_config_str()}}}_bsa_config={{{da_config.bsa_config}}}'

    DATABASE_ROOT = get_global_var('DATABASE_ROOT')
    INTRA_BSA_EXE_PLANS_DIR = f'{DATABASE_ROOT}/{CLUSTER_NAME}/{PLATFORM}/intra_bsa_exe_plans'
    INTRA_BSA_EXE_PLANS_KV = f'{DATABASE_ROOT}/intra_bsa_exe_plans_kv.json'
    with open(f'{INTRA_BSA_EXE_PLANS_KV}', 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    
    key_suffixes = []
    for KERNEL_SCHEDULE_TYPE in ['ILP', 'Flexflow']:
        for KERNEL_TILE_TYPE in ['w/o_kernel_tile', 'w_kernel_tile']:
            key_suffixes.append(f'_ablation=({KERNEL_TILE_TYPE},{KERNEL_SCHEDULE_TYPE})')
    for key_suffix in key_suffixes:
        key = f'{key_preffix}{key_suffix}'
        # load exe_plan
        plan_id = intra_bsa_exe_plans_dict[key]
        with open(f'{INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
            intra_bsa_execution_plan: Execution_Plan = pickle.load(fin)
        # inter_comp_plans_dict: {intra_bsa_key: intra_bsa_exe_plan}
        inter_comp_plans_dict = {
            ((1, 1), str(da_config.bsa_config.bsa_repr)): intra_bsa_execution_plan,
        }
        inter_comp_plans_dicts.append(inter_comp_plans_dict)
    
    # Execution:
    # 1 baselines
    # [TODO]
    
    # 2 orchestrated_attn_func:
    # [TODO]: check corretness of da_config✅&exp_configs✅
    benchmark_op = partial(benchmark_orchestrate_bsa,
        args, orchestrated_attn_func, da_config, tensor_buf, log=True, exp_configs=[exp_config], 
        global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
        warmup=WARMUP, num_iter=NUM_ITER, mode='profile', inter_comp_plans_dicts=inter_comp_plans_dicts,
    )
    benchmark_op(use_cudagraph=False) 
    
    return

    for fob in fobs:
        inter_ablation_suffixes = ['']            # for SP0 = 1, i.e. profile
        plan_types = ['automatic']  # for profile is OK
        exp_configs = []
        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/{CLUSTER_NAME}/{PLATFORM}/inter_SP{node_num}_fob={fob}' # Dummy !!!
        plan_paths = []
    
        # Create exp_configs
        dummy_da_config = Dist_Attn_Config(SP=SPs, S=(256 * world_size, 256 * world_size), Nh=(1, 1), D=D, bs=bs, causal=True)
        plan_name_prefix = dummy_da_config.get_plan_name(fob=fob)
        for suffix in inter_ablation_suffixes:
            plan_paths.append(f'{par_dir}/{plan_name_prefix}{suffix}.pkl')
        for plan_type in plan_types:
            for plan_path in plan_paths:
                exp_config = Evaluation_Configs(
                        plan_type=plan_type,
                        MAX_QUEUE_SIZE=0,
                        fob=fob,
                        plan_path=plan_path,
                        inter_comp_profile_map=inter_comp_profile_map,
                    )
                exp_configs.append(exp_config)
        # print_rank_0(f'plan_type: {plan_type}')
        # print_rank_0(f'plan_paths: {plan_paths}')
        # print_rank_0(f'exp_configs: {exp_configs}')
        # for causal in causals:
        for bsa_config in BSA_configs:
            S_base = [1 << logS for logS in range(int(math.log2(S_BOUND[0])), int(math.log2(S_BOUND[1])) + 1)]
            multiplying_powers = [1]
            Sqkvs = [S * power for S in S_base for power in multiplying_powers if S * power <= S_BOUND[1]]
            Sqkvs = sorted(list(set(Sqkvs)))    # Sq per GPU
            Sqs = Skvs = Sqkvs
            # # [Debug]: da_config: SP=(1,8),Sg=(1280,28672),S=(10240,229376),Nh=(1,1),bs=1,D=128,causal=False,hierarchy=1:    Error
            # Sqs = [1280]
            # Skvs = [28672]
            # Nhs = [1]
            # Sqs = [327680 // world_size]
            # Sqkvs = [256, 512]  # for debug
            # Sqkvs = [1 * 1024, 2 * 1024, 4 * 1024, 8 * 1024]  # for debug
            # Sqkvs = [1 * 1024]  # for debug
            # Sqkvs = [64 * 1024]
            # print_rank_0(f'Sqkvs: {Sqkvs}')
            # Sqs, Skvs = [256], [49152]    # 256,49152
            print_rank_0(f'Sqs: {Sqs}')
            print_rank_0(f'Skvs: {Skvs}')
            print_rank_0(f'fob={fob}')  # [NOTE]: Necessary
                  
            for Nh in Nhs:
                for Sq in Sqs: # S per GPU
                    for Skv in Skvs: # S per GPU
                        if Sq != Skv:
                            continue
                        # print(f'bsa_config: {bsa_config}')
                        da_config = Dist_Attn_Config(SP=SPs, S=(Sq * world_size, Skv * world_size), Nh=(Nh, Nh), D=D, bs=bs, causal=True, bsa_config=bsa_config)
                        print_rank_0(f'da_config: {da_config}')
                        
                        # Create all inter_comp_plans for profiling for BSA
                        inter_comp_plans_dicts = []
                        # 2 * 2 = 4 types (2: ILP, Flexflow; 2: non-fused, fused)
                        intra_ablation_suffixes = ['', '_fused', '_ablation1', '_fused_ablation1']
                        # intra_ablation_suffixes = ['', '_ablation1']
                        # intra_ablation_suffixes = ['']  # for torch.profiler
                        # intra_ablation_suffixes = []
                        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{local_size}_fob={fob}_{bsa_config}'
                        # print(f'BSA par_dir: {par_dir}')
                        plan_name_prefix = da_config.get_plan_name(fob=fob)
                        # load intra plans
                        for suffix in intra_ablation_suffixes:
                            plan_path = f'{par_dir}/{plan_name_prefix}{suffix}.pkl'
                            with open(plan_path, 'rb') as fin:
                                cur_execution_plan: Execution_Plan = pickle.load(fin)
                                inter_comp_plans_dict = {
                                    ((1, 1), bsa_config.to_string()): cur_execution_plan,
                                }
                                # print_rank_0(f'plan_path: {plan_path}')
                                # if rank == 0:
                                #     cur_execution_plan.print_lp_result()
                                inter_comp_plans_dicts.append(inter_comp_plans_dict)
                    
                        # assert len(inter_comp_plans_dicts) == 4 if (causal or (Nh > 1)) else 8
                        
                        # Execution:
                        # 1 baselines
                        # None
                        
                        # 2 orchestrated_attn_func:
                        benchmark_op = partial(benchmark_orchestrate_bsa,
                            args, orchestrated_attn_func, da_config, tensor_buf, log=True, exp_configs=exp_configs, 
                            global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
                            warmup=WARMUP, num_iter=NUM_ITER, mode='profile', inter_comp_plans_dicts=inter_comp_plans_dicts,
                        )
                        benchmark_op(use_cudagraph=False)          

def main():
    ncclcomm_global, gloo_global_group = initialize_distribution()
    
    # Step0: top-> down; need only 1 cpu; (w/o cache/bypass)✅
    if torch.distributed.get_rank() == 0:
        global_bsa_configs, intra_node_bsa_configs, shape_configs = get_configs()
        exp_configs = get_exp_configs()
        if isinstance(exp_configs, Evaluation_Configs):
            exp_configs = [exp_configs]
    #   [NOTE]: total exp space is (global_bsa_configs/intra_node_bsa_configs) x shape_configs x exp_configs
    
    # Step1: Generate the intra-BSA; need all cpus on one node; (w cache/bypass)
    #   Initialize Profile_DataBase
    intra_plan_id = 0
    intra_da_configs: List[Dist_Attn_Config] = []
    if torch.distributed.get_rank() == 0:
        prof_db = Prof_DB()
        for exp_config in exp_configs:
            for intra_node_bsa_config in intra_node_bsa_configs:
                for Nh in shape_configs['Nhs']:
                    for S in shape_configs['Ss']:
                        for bs in shape_configs['BSs']:
                            for D in shape_configs['Ds']:
                                shape_config = {
                                    'Nh': (Nh, Nh),
                                    'S': (S, S),
                                    'bs': bs,
                                    'D': D,
                                }
                                da_config = Dist_Attn_Config.from_bsa_config(
                                    intra_node_bsa_config, 
                                    shape_config=shape_config,
                                    hierarchy=1,
                                )
                                intra_da_configs.append(da_config)
                                print(f'intra_plan_id: {intra_plan_id}', flush=True)
                                generate_intra_execution_plans(exp_config, da_config, prof_db)
                                intra_plan_id += 1
                                
    torch.distributed.barrier()
    
    # Step2: Profile all BSA at intra_SP=8; one node, one processor occupies one gpu and even cpus; (w cache/bypass)
    if torch.cuda.is_available():
        MAX_S, MAX_NH, MAX_D, MAX_bs = 0, 0, 0, 0
        for da_config in intra_da_configs:
            MAX_S = max(MAX_S, max(da_config.shape_config['S']))
            MAX_NH = max(MAX_NH, max(da_config.shape_config['Nh']))
            MAX_D = max(MAX_D, da_config.shape_config['D'])
            MAX_bs = max(MAX_bs, da_config.shape_config['bs'])
        print_rank_0(f'MAX_S={MAX_S}; MAX_NH={MAX_NH}; MAX_D={MAX_D}; MAX_bs={MAX_bs}')
        print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')

        tensor_buf = torch.empty(
            (MAX_bs * MAX_S * MAX_NH * MAX_D * 4) * 3                       # k, v, dk, dv
        + (MAX_bs * MAX_S * MAX_NH * (MAX_D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
        + (MAX_bs * MAX_S * MAX_NH * (MAX_D * 2) + (1 * (2 + 1))) * 2,    # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
            device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
        )   # 6 * 512MB = 3GB
        args = parse_args()
        for exp_config in exp_configs:  # fobs
            for da_config in intra_da_configs:
                profile_all_intra_BSA(args, exp_config, da_config, ncclcomm_global, gloo_global_group, tensor_buf)
    
    # Step3: Generate execution plans for all BSA at inter_SP=2,4,8; need all cpus on one node; (w cache/bypass)  [TODO]
    pass

    
    
if __name__ == '__main__':
    main()