import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Evaluation_Configs
import numpy as np
from search_algo.bsa_config import BSA_Repr, BSA_Config
from search_algo.bsa_utils import create_bsa_block_table, split_to_node_configs
from search_algo.utils import combine_list_to_0, convert_block_table_to_value, print_rank_0

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
    # [TODO]: How to set configs ???
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
            (8, 8),
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
    Ss_per_node = [
        64 * 1024,
    ]    # S per node
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
    # # 1. BSA_Repr for stride(1/16, 4, 3) (after remapping)
    # global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(0), cmap=cmap_finest))
    # 2. BSA_Repr for local+global(1/16, 1, 1) (no remapping)
    global_bsa_reprs.append(BSA_Repr(block_table=create_bsa_block_table(1), cmap=cmap_finest))
    # Create inter_node_bsa_configs from global_bsa_reprs   ✅
    # break global_bsa_reprs down to node-level bsa_repr&bsa_configs !!! ✅
    inter_node_bsa_configs = []
    # intra_node_bsa_configs = set()
    intra_node_bsa_configs = []
    for i, gbr in enumerate(global_bsa_reprs):
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
            # inter_node_bsa_configs[-1][CP] = inter_node_bsa_config
            inter_node_bsa_configs.append(inter_node_bsa_config)
            node_bsa_configs = split_to_node_configs(inter_node_bsa_config)
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
        print_rank_0(f'CP: {intra_node_bsa_config.CP}, minimum_Par_D: {intra_node_bsa_config.bsa_repr.minimum_Par_D}\n{block_table_value}')
    intra_node_shape_configs = {
        'Nhs': Nhs,
        'Ss': Ss_per_node,
        'BSs': [bs],
        'Ds': [D],
    }
    return inter_node_bsa_configs, intra_node_bsa_configs, intra_node_shape_configs

def step0_top_down_decompose():
    # Step0: top-> down; need only 1 cpu; (w/o cache/bypass)✅
    inter_node_bsa_configs, intra_node_bsa_configs, shape_configs = get_configs()
    exp_configs = get_exp_configs()
    if isinstance(exp_configs, Evaluation_Configs):
        exp_configs = [exp_configs]
    #   [NOTE]: total exp space is (inter_node_bsa_configs/intra_node_bsa_configs) x shape_configs x exp_configs
    return inter_node_bsa_configs, intra_node_bsa_configs, shape_configs, exp_configs
