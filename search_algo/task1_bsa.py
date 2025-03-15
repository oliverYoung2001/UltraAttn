import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, \
                                      get_profile_data, get_init_schedule_list, get_cc_optimal_schedule, get_cc_optimal_schedule_from_table
from search_algo.dependent_graph import Dependent_Graph
from search_algo.graph_transformation_engine import Graph_Transformation_Engine
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from search_algo.bsa_config import BSA_Repr, BSA_Config
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs
from search_algo.utils import combine_list_to_0, convert_block_table_to_value
from search_algo.database import Prof_DB
import torch
import torch.distributed as dist
from functools import partial
import socket
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from search_algo.workload_partition import solve_sparse_from_bsa

PLATFORM = os.getenv(f'PLATFORM')

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
        
    else:
        raise Exception("Unknown Launcher !!!")
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
    global PROC_INFO
    PROC_INFO = get_proc_info()
    # print(f'PROC_INFO: {PROC_INFO}')
    
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    # MASTER_ADDR = 'localhost'
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    dist.init_process_group(
        backend="nccl", 
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

    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    torch.cuda.set_device(device)
    
    # preprocess placeholder_op
    global placeholder_op
    SYNC_SIZE = 8 * pow(1024, 3) # 8GB
    sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=device)
    placeholder_op = partial(ncclcomm_global.all_reduce, sync_tensor)

def get_exp_configs():
    # plan_type = 'automatic'
    plan_type = 'manual'  # for noncausal !!!
    # plan_type = 'ablation1'
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

    Ss = [
        # 256, 
        # 512, 
        # 1 * 1024, 
        # 2 * 1024, 
        # 4 * 1024, 
        # 8 * 1024, 
        # 16 * 1024, 
        # 32 * 1024,    # fused failed on 8 * 8 !!!
        64 * 1024,    # fused failed on 4 * 8 !!!
    ]    # S per GPU
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

def generate_intra_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, prof_db: Prof_DB):
    # [TODO]: 
    exp_config.hierarchy = da_config.hierarchy = 1
    m_config = get_profile_data(da_config.SP, exp_config.hierarchy)
    prof_db.update_m_config(m_config)
    # Calc optimal schedule !
    cc_optimal_schedule = get_cc_optimal_schedule_from_table(da_config, m_config, solve_sparse_from_bsa(da_config.bsa_config))
    # End
    # cc_optimal_schedule = get_cc_optimal_schedule(da_config, m_config)  # Dist_Attn_Schedule(np.array)
    # [TODO]: replace `hardcode with `workload_partition.py`
    
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)

    if da_config.bsa_config:
        print(f'bsa_config_string: {da_config.bsa_config.to_string()}')
        par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{da_config.hierarchy_sp}_fob={exp_config.fob}_{da_config.bsa_config}'
    else:
        par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{da_config.hierarchy_sp}_fob={exp_config.fob}_causal={da_config.causal}'
    print(f'par_dir: {par_dir}')
    os.makedirs(par_dir, exist_ok=True)
    # return
    plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    for plan_type in plan_types:
        # Raw Execution_Plan:
        print(f'Raw, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True)
        # if not plan_type == 'automatic':
        if True:
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type)
            execute_plan.print_lp_result()
            plan_name = execute_plan.get_plan_name()
            if plan_type == 'ablation1':
                plan_name = f'{plan_name}_{plan_type}'
            # Dump Execution_Plan:
            plan_file = f'{par_dir}/{plan_name}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        
        # Transformed Execution_Plans:
        print(f'Fused, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True)
        gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
        execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=plan_type)
        if execute_plan is None:    # No feasible transformations
            continue
        assert isinstance(execute_plan, Execution_Plan)
        plan_name = f'{execute_plan.get_plan_name()}_fused'
        if plan_type == 'ablation1':
            plan_name = f'{plan_name}_{plan_type}'
        # Dump Execution_Plan:
        plan_file = f'{par_dir}/{plan_name}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)

def main():
    initialize_distribution()
    
    # Step0: top-> down; need only 1 cpu; (w/o cache/bypass)✅
    if torch.distributed.get_rank() == 0:
        global_bsa_configs, intra_node_bsa_configs, shape_configs = get_configs()
        exp_configs = get_exp_configs()
        if isinstance(exp_configs, Evaluation_Configs):
            exp_configs = [exp_configs]
    #   [NOTE]: total exp space is (global_bsa_configs/intra_node_bsa_configs) x shape_configs x exp_configs
    # Step1: Generate the intra-BSA; need all cpus on one node; (w cache/bypass)
    #   Initialize Profile_DataBase
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
                                generate_intra_execution_plans(exp_config, da_config, prof_db)
    # [TODO]: sync all ranks
    # [TODO]: prepare prof_db with cache on dick !!!
    
    # Step2: Profile all BSA at intra_SP=8; one node, one processor occupies one gpu and even cpus; (w cache/bypass)
    # Step3: Generate execution plans for all BSA at inter_SP=2,4,8; need all cpus on one node; (w cache/bypass)  [TODO]
    pass

    
    
if __name__ == '__main__':
    main()