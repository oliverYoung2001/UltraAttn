import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Evaluation_Configs
import numpy as np
from search_algo.bsa_config import BSA_Repr, BSA_Config
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs
from search_algo.utils import combine_list_to_0, convert_block_table_to_value, print_rank_0
import math

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

def get_bsa_configs():
    PLATFORM = os.getenv(f'PLATFORM')
    CPs = [
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
    ]
    Ss_total = [
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1 * 1024 * 1024,
        2 * 1024 * 1024,    # 2M
    ]
    S_per_gpu_LB = 256
    S_per_gpu_UB = 64 * 1024   # (8, 8) -> 2M; aims to limit memory usage
    GPU_PER_NODE = 8
    Ss_per_node = [1 << i for i in range(math.ceil(math.log2(Ss_total[0] // CPs[-1][-1])), \
                                        math.ceil(math.log2(S_per_gpu_UB * GPU_PER_NODE)) + 1)]
    
    Nhs = [
        1,
        32,
    ]
    bs = 1
    D = 128
    
    cmap_finest = None
    global_bsa_reprs = []
    # 1. BSA_Repr for stride(1/16, 4, 3) (after remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(0), cmap=cmap_finest))
    # 2. BSA_Repr for local+global(1/16, 1, 1) (no remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(1), cmap=cmap_finest))
    # Create inter_node_bsa_configs from global_bsa_reprs   ✅
    # break global_bsa_reprs down to node-level bsa_repr&bsa_configs !!! ✅
    inter_node_bsa_configs = []
    # intra_node_bsa_configs = set()
    intra_node_bsa_configs = []
    for i, gbr in enumerate(global_bsa_reprs):  # [NOTE]: Assume no 'kernel tile' at inter-level
        # inter_node_bsa_configs.append({})
        for CP in CPs:
            # print_rank_0(f'CP: {CP}')
            # print_rank_0(f'gbr.minimum_Par_D: {gbr.minimum_Par_D}')    # 16
            inter_node_bsa_config = BSA_Config(
                None, None,
                {
                    'bsa_repr': gbr,
                    'CP': CP,
                }
            )
            inter_node_bsa_configs.append(inter_node_bsa_config)
            node_bsa_configs = split_to_node_configs(inter_node_bsa_config)
            combine_list_to_0(intra_node_bsa_configs, node_bsa_configs)
    
    # filter out all empty configs
    filtered_configs = []
    for c in intra_node_bsa_configs:
        if not c.bsa_repr.check_empty():
            filtered_configs.append(c)
    intra_node_bsa_configs = filtered_configs   # [NOTE]: All intra_node_bsa_config share the same Ss_per_node !!!
    for intra_node_bsa_config in intra_node_bsa_configs:
        block_table_raw_value = convert_block_table_to_value(intra_node_bsa_config.block_table_raw)
        print_rank_0(f'CP: {intra_node_bsa_config.CP}, minimum_Par_D: {intra_node_bsa_config.bsa_repr.minimum_Par_D}\n{block_table_raw_value}')
    
    inter_node_shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss_total,
        'BSs': [bs],
        'Ds': [D],
    }
    intra_node_shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss_per_node,
        'BSs': [bs],
        'Ds': [D],
    }
    shape_config_dict = {
        'inter': inter_node_shape_configs,
        'intra': intra_node_shape_configs,
        'S_per_gpu_BOUND': (S_per_gpu_LB, S_per_gpu_UB),
    }
    return inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict

def get_bsa_configs_debug():
    PLATFORM = os.getenv(f'PLATFORM')
    CPs = [
        (8, 1),
    ]
    Ss_total = [
        8 * 1024,
    ]
    S_per_gpu_LB = 256
    S_per_gpu_UB = 64 * 1024   # (8, 8) -> 2M; aims to limit memory usage
    GPU_PER_NODE = 8
    Ss_per_node = [1 << i for i in range(math.ceil(math.log2(min(Ss_total) // CPs[-1][-1])), \
                                        math.ceil(math.log2(min(
                                            max(Ss_total) // CPs[0][-1], S_per_gpu_UB * GPU_PER_NODE)
                                        )) + 1)]
    print_rank_0(f'Ss_per_node: {Ss_per_node}')
    Nhs = [
        # 1,
        32,
    ]
    bs = 1
    D = 128
    
    cmap_finest = None
    global_bsa_reprs = []
    # # 1. BSA_Repr for stride(1/16, 4, 3) (after remapping)
    # global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(0), cmap=cmap_finest))
    # 2. BSA_Repr for local+global(1/16, 1, 1) (no remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(1), cmap=cmap_finest))
    # Create inter_node_bsa_configs from global_bsa_reprs   ✅
    # break global_bsa_reprs down to node-level bsa_repr&bsa_configs !!! ✅
    inter_node_bsa_configs = []
    # intra_node_bsa_configs = set()
    intra_node_bsa_configs = []
    for i, gbr in enumerate(global_bsa_reprs):  # [NOTE]: Assume no 'kernel tile' at inter-level
        # inter_node_bsa_configs.append({})
        for CP in CPs:
            # print_rank_0(f'CP: {CP}')
            # print_rank_0(f'gbr.minimum_Par_D: {gbr.minimum_Par_D}')    # 16
            inter_node_bsa_config = BSA_Config(
                None, None,
                {
                    'bsa_repr': gbr,
                    'CP': CP,
                }
            )
            inter_node_bsa_configs.append(inter_node_bsa_config)
            node_bsa_configs = split_to_node_configs(inter_node_bsa_config)
            combine_list_to_0(intra_node_bsa_configs, node_bsa_configs)
    
    # filter out all empty configs
    filtered_configs = []
    for c in intra_node_bsa_configs:
        if not c.bsa_repr.check_empty():
            filtered_configs.append(c)
    intra_node_bsa_configs = filtered_configs   # [NOTE]: All intra_node_bsa_config share the same Ss_per_node !!!
    for intra_node_bsa_config in intra_node_bsa_configs:
        block_table_raw_value = convert_block_table_to_value(intra_node_bsa_config.block_table_raw)
        print_rank_0(f'CP: {intra_node_bsa_config.CP}, minimum_Par_D: {intra_node_bsa_config.bsa_repr.minimum_Par_D}\n'
                     f'{block_table_raw_value}')
    
    inter_node_shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss_total,
        'BSs': [bs],
        'Ds': [D],
    }
    intra_node_shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss_per_node,
        'BSs': [bs],
        'Ds': [D],
    }
    shape_config_dict = {
        'inter': inter_node_shape_configs,
        'intra': intra_node_shape_configs,
        'S_per_gpu_BOUND': (S_per_gpu_LB, S_per_gpu_UB),
    }
    return inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict

def step0_top_down_decompose():
    # Step0: top-> down; need only 1 cpu; (w/o cache/bypass)✅
    # inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs()
    inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict = get_bsa_configs_debug()
    exp_configs = get_exp_configs()
    if isinstance(exp_configs, Evaluation_Configs):
        exp_configs = [exp_configs]
    #   [NOTE]: total exp space is (inter_node_bsa_configs/intra_node_bsa_configs) x shape_configs x exp_configs
    return inter_node_bsa_configs, intra_node_bsa_configs, shape_config_dict, exp_configs
