import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import numpy as np
from search_algo.global_vars import *

def get_qo_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool, da_config=None):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    qo_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    qo_schedule_table.fill(TASK_STATUS.EMPTY.value) 
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    qo_schedule_table[i, j, k, l] = S_map[i, l]
    return qo_schedule_table

def get_kv_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool, da_config=None):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    kv_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    kv_schedule_table -= 1  # -1 means not used
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    kv_schedule_table[i, j, k, l] = S_map[i, k]
    return kv_schedule_table

def get_cc_optimal_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool, da_config=None):
    assert len(split_degrees) == 4  # (split_Sq, split_Skv, split_bs, split_Nh)
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    assert split_degrees[0] == split_degrees[1]
    assert split_degrees[2] == split_degrees[3] == 1
    # print(f'split_degrees: {split_degrees}, causal: {split_degrees}', flush=True)
    cc_schedule_table = None    # (split_bs, split_Nh, split_Sq, split_Skv)
    if da_config.bsa_config:    # BSA
        bsa_config = da_config.bsa_config
        # stride_16_4_3 (8x2)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2},
        
                # 0  1  0  /  /  /  /  /  
                # 2  1  2  1  /  /  /  /  
                # 2  3  2  3  /  /  /  /  
                # /  3  0  3  /  /  /  /  
                # /  /  /  /  4  6  6  /  
                # /  /  /  /  7  5  5  7  
                # /  /  /  /  4  6  6  4  
                # /  /  /  /  /  5  5  7  
                # [0, 1, 0, -1, -1, -1, -1, -1],
                # [2, 1, 2, 1, -1, -1, -1, -1],
                # [2, 3, 2, 3, -1, -1, -1, -1],
                # [-1, 3, 0, 3, -1, -1, -1, -1],
                # [-1, -1, -1, -1, 4, 6, 6, -1],
                # [-1, -1, -1, -1, 7, 5, 5, 7],
                # [-1, -1, -1, -1, 4, 6, 6, 4],
                # [-1, -1, -1, -1, -1, 5, 5, 7],
                

        # stride_16_4_3 (8x4)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 1},
                
                # 0  2  2  0  2  6  /  /  
                # 0  1  4  0  4  1  /  /  
                # 0  1  2  0  2  1  1  0  
                # 3  2  2  3  3  6  6  6  
                # 7  4  4  7  4  1  1  7  
                # 7  4  5  7  4  5  5  7  
                # /  /  5  3  3  6  6  6  
                # /  /  5  3  3  5  5  7  
                # [[ 0,  2,  2,  0,  2,  6, -1, -1],
                # [ 0,  1,  4,  0,  4,  1, -1, -1],
                # [ 0,  1,  2,  0,  2,  1,  1,  0],
                # [ 3,  2,  2,  3,  3,  6,  6,  6],
                # [ 7,  4,  4,  7,  4,  1,  1,  7],
                # [ 7,  4,  5,  7,  4,  5,  5,  7],
                # [-1, -1,  5,  3,  3,  6,  6,  6],
                # [-1, -1,  5,  3,  3,  5,  5,  7]]

        # stride_16_4_3 (8x8)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
#         0  1  2  0  0  1  1  2  
#         0  1  5  0  0  1  1  5  
#         2  7  2  6  6  7  6  2  
#         4  7  3  3  4  7  3  7  
#         4  4  5  3  4  5  3  5  
#         4  4  5  6  4  5  6  5  
#         2  1  2  6  6  1  6  2  
#         0  7  3  3  0  7  3  7  
        
# [[0, 1, 2, 0, 0, 1, 1, 2],
#  [0, 1, 5, 0, 0, 1, 1, 5],
#  [2, 7, 2, 6, 6, 7, 6, 2],
#  [4, 7, 3, 3, 4, 7, 3, 7],
#  [4, 4, 5, 3, 4, 5, 3, 5],
#  [4, 4, 5, 6, 4, 5, 6, 5],
#  [2, 1, 2, 6, 6, 1, 6, 2],
#  [0, 7, 3, 3, 0, 7, 3, 7]]

        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 2), 'global_blocks': (0, 0), 'replicate': 1},
#         0  1  4  3  0  1  3  4  
#         7  1  2  3  7  1  3  2  
#         0  1  2  6  0  1  6  2  
#         0  5  2  3  0  5  3  2  
#         /  /  /  /  4  6  6  4  
#         /  /  /  /  4  5  5  4  
#         /  /  /  /  7  6  6  7  
#         /  /  /  /  7  5  5  7  
# [[ 0,  1,  4,  3,  0,  1,  3,  4],
#  [ 7,  1,  2,  3,  7,  1,  3,  2],
#  [ 0,  1,  2,  6,  0,  1,  6,  2],
#  [ 0,  5,  2,  3,  0,  5,  3,  2],
#  [-1, -1, -1, -1,  4,  6,  6,  4],
#  [-1, -1, -1, -1,  4,  5,  5,  4],
#  [-1, -1, -1, -1,  7,  6,  6,  7],
#  [-1, -1, -1, -1,  7,  5,  5,  7]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 1), 'global_blocks': (0, 0), 'replicate': 1},
# 0  2  2  0  /  /  /  /  
# 0  1  1  0  /  /  /  /  
# 0  2  2  0  /  /  /  /  
# 3  1  1  3  /  /  /  /  
# 7  5  4  4  4  5  5  7  
# 7  5  4  4  4  5  5  7  
# 3  2  2  3  6  6  6  3  
# 7  1  1  3  6  6  6  7  
# [[ 0,  2,  2,  0, -1, -1, -1, -1],
#  [ 0,  1,  1,  0, -1, -1, -1, -1],
#  [ 0,  2,  2,  0, -1, -1, -1, -1],
#  [ 3,  1,  1,  3, -1, -1, -1, -1],
#  [ 7,  5,  4,  4,  4,  5,  5,  7],
#  [ 7,  5,  4,  4,  4,  5,  5,  7],
#  [ 3,  2,  2,  3,  6,  6,  6,  3],
#  [ 7,  1,  1,  3,  6,  6,  6,  7]]
        # lg_16_1_1 (8x2)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
# [[ 0, -1, -1, -1, -1, -1, -1, -1],
#  [ 1, -1, -1, -1, -1, -1, -1, -1],
#  [ 2, -1, -1, -1, -1, -1, -1, -1],
#  [ 3, -1, -1, -1, -1, -1, -1, -1],
#  [ 4, -1, -1, -1, -1, -1, -1, -1],
#  [ 5, -1, -1, -1, -1, -1, -1, -1],
#  [ 6, -1, -1, -1, -1, -1, -1, -1],
#  [ 7, -1, -1, -1, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
# [[ 0,  1,  2,  3,  4,  5,  6,  7],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
# [[ 0, -1, -1, -1, -1, -1, -1, -1],
#  [-1,  1, -1, -1, -1, -1, -1, -1],
#  [-1, -1,  2, -1, -1, -1, -1, -1],
#  [-1, -1, -1,  3, -1, -1, -1, -1],
#  [-1, -1, -1, -1,  4, -1, -1, -1],
#  [-1, -1, -1, -1, -1,  5, -1, -1],
#  [-1, -1, -1, -1, -1, -1,  6, -1],
#  [-1, -1, -1, -1, -1, -1, -1,  7]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
# [[ 0,  0,  2,  3,  3,  1,  2,  1],
#  [ 5,  1, -1, -1, -1, -1, -1, -1],
#  [ 6, -1,  2, -1, -1, -1, -1, -1],
#  [ 5, -1, -1,  3, -1, -1, -1, -1],
#  [ 0, -1, -1, -1,  4, -1, -1, -1],
#  [ 7, -1, -1, -1, -1,  5, -1, -1],
#  [ 6, -1, -1, -1, -1, -1,  6, -1],
#  [ 7, -1, -1, -1, -1, -1, -1,  7]]
        # lg_16_1_1 (8x4)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
# [[ 0,  0, -1, -1, -1, -1, -1, -1],
#  [ 1,  1, -1, -1, -1, -1, -1, -1],
#  [ 2,  6, -1, -1, -1, -1, -1, -1],
#  [ 7,  4, -1, -1, -1, -1, -1, -1],
#  [ 2,  4, -1, -1, -1, -1, -1, -1],
#  [ 5,  3, -1, -1, -1, -1, -1, -1],
#  [ 5,  3, -1, -1, -1, -1, -1, -1],
#  [ 7,  6, -1, -1, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
# [[ 0,  6,  5,  5,  6,  4,  0,  4],
#  [ 3,  1,  1,  7,  2,  3,  7,  2],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
# [[ 0,  1, -1, -1, -1, -1, -1, -1],
#  [ 0,  1, -1, -1, -1, -1, -1, -1],
#  [-1, -1,  2,  3, -1, -1, -1, -1],
#  [-1, -1,  2,  3, -1, -1, -1, -1],
#  [-1, -1, -1, -1,  4,  5, -1, -1],
#  [-1, -1, -1, -1,  4,  5, -1, -1],
#  [-1, -1, -1, -1, -1, -1,  6,  7],
#  [-1, -1, -1, -1, -1, -1,  6,  7]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
# [[ 0,  1,  4,  0,  4,  4,  1,  0],
#  [ 0,  1,  3,  3,  3,  1,  1,  0],
#  [ 2,  5,  2,  5, -1, -1, -1, -1],
#  [ 2,  7,  3,  3, -1, -1, -1, -1],
#  [ 6,  5, -1, -1,  4,  4, -1, -1],
#  [ 2,  5, -1, -1,  2,  5, -1, -1],
#  [ 6,  7, -1, -1, -1, -1,  6,  6],
#  [ 6,  7, -1, -1, -1, -1,  7,  7]]
        # lg_16_1_1 (8x8)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
# [[0, 1, 2, 0, 0, 1, 1, 2],
#  [0, 1, 5, 0, 0, 1, 1, 5],
#  [2, 7, 2, 6, 6, 7, 6, 2],
#  [4, 7, 3, 3, 4, 7, 3, 7],
#  [4, 4, 5, 3, 4, 5, 3, 5],
#  [4, 4, 5, 6, 4, 5, 6, 5],
#  [2, 1, 2, 6, 6, 1, 6, 2],
#  [0, 7, 3, 3, 0, 7, 3, 7]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
# [[ 0,  5,  4,  0, -1, -1, -1, -1],
#  [ 6,  1,  1,  7, -1, -1, -1, -1],
#  [ 2,  5,  2,  7, -1, -1, -1, -1],
#  [ 6,  3,  4,  3, -1, -1, -1, -1],
#  [ 0,  5,  4,  0, -1, -1, -1, -1],
#  [ 2,  5,  2,  7, -1, -1, -1, -1],
#  [ 6,  3,  4,  3, -1, -1, -1, -1],
#  [ 6,  1,  1,  7, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
# [[ 0,  4, -1,  5,  4,  5,  7,  4],
#  [ 1,  1,  2,  3,  6,  3,  2,  6],
#  [ 1,  1,  2,  5,  6,  5,  2,  6],
#  [ 0,  4, -1,  3,  7,  3,  7,  7],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1],
#  [-1, -1, -1, -1, -1, -1, -1, -1]]
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
# [[ 0,  2,  2,  0, -1, -1, -1, -1],
#  [ 3,  1,  1,  3, -1, -1, -1, -1],
#  [ 0,  2,  2,  0, -1, -1, -1, -1],
#  [ 3,  1,  1,  3, -1, -1, -1, -1],
#  [-1, -1, -1, -1,  4,  6,  6,  4],
#  [-1, -1, -1, -1,  4,  5,  5,  4],
#  [-1, -1, -1, -1,  7,  6,  6,  7],
#  [-1, -1, -1, -1,  7,  5,  5,  7]]

        # stride_16_4_3 (8x2)
        if bsa_config == {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2}:
            cc_schedule_table = np.array([[[
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
            ]]], dtype=np.int32)
        if bsa_config == 

    else:   # causal or full
        if split_degrees[0] == 16:
            cc_schedule_table = np.array([[[
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
                [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
                [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
                [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
                [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
                [12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15],
                [12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15],
                [12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15],
                [12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15],
            ]]], dtype=np.int32)
            if causal:
                for i in range(split_degrees[0]):
                    for j in range(split_degrees[1]):
                        if i < j:
                            cc_schedule_table[0, 0, i, j] = - 1
                return cc_schedule_table
            else:
                return cc_schedule_table
        if split_degrees[0] == 8:
            cc_schedule_table = np.array([[[    # 2 * 4 = Y * X
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
                [4, 5, 6, 7, 4, 5, 6, 7],
            ]]], dtype=np.int32)
            if causal:
                for i in range(split_degrees[0]):
                    for j in range(split_degrees[1]):
                        if i < j:
                            cc_schedule_table[0, 0, i, j] = - 1
                # return cc_schedule_table
                cc_schedule_tables = []
                cc_schedule_table = np.array([[[
                    [ 0, -1, -1, -1, -1, -1, -1, -1],
                    [ 0,  1, -1, -1, -1, -1, -1, -1],
                    [ 1,  1,  2, -1, -1, -1, -1, -1],
                    [ 1,  1,  3,  3, -1, -1, -1, -1],
                    [ 0,  4,  0,  0,  4, -1, -1, -1],
                    [ 2,  4,  2,  5,  4,  5, -1, -1],
                    [ 2,  6,  2,  5,  7,  5,  6, -1],
                    [ 7,  6,  3,  3,  7,  3,  6,  7],
                ]]], dtype=np.int32)
                cc_schedule_tables.append(cc_schedule_table)
                return cc_schedule_tables
            else:
                return cc_schedule_table
        if split_degrees[0] == 6:
            if causal:
                cc_schedule_tables = []
                cc_schedule_table = np.array([[[
                    [0, -1, -1, -1, -1, -1],
                    [1,  1, -1, -1, -1, -1],
                    [0,  0,  2, -1, -1, -1],
                    [1,  3,  2,  3, -1, -1],
                    [0,  4,  5,  3,  4, -1],
                    [1,  4,  5,  3,  4,  5],
                ]]], dtype=np.int32)
                cc_schedule_tables.append(cc_schedule_table)
                cc_schedule_table = np.array([[[
                    [0, -1, -1, -1, -1, -1],
                    [1,  1, -1, -1, -1, -1],
                    [0,  2,  2, -1, -1, -1],
                    [0,  3,  3,  3, -1, -1],
                    [0,  2,  2,  5,  4, -1],
                    [1,  1,  4,  5,  4,  5],
                ]]], dtype=np.int32)
                cc_schedule_tables.append(cc_schedule_table)
                return cc_schedule_tables
            else:
                raise NotImplementedError()
        if split_degrees[0] == 4:
            if causal:
                cc_schedule_tables = []
                # cc_schedule_table = np.array([[[
                #     [0, -1, -1, -1],
                #     [1,  1, -1, -1],
                #     [0,  0,  2, -1],
                #     [1,  3,  2,  3],
                # ]]], dtype=np.int32)
                # cc_schedule_tables.append(cc_schedule_table)
                cc_schedule_table = np.array([[[
                    [0, -1, -1, -1],
                    [1,  1, -1, -1],
                    [0,  0,  2, -1],
                    [3,  1,  2,  3],
                ]]], dtype=np.int32)
                cc_schedule_tables.append(cc_schedule_table)
                # cc_schedule_table = np.array([[[
                #     [0, -1, -1, -1],
                #     [1,  1, -1, -1],
                #     [1,  2,  2, -1],
                #     [0,  3,  0,  3],
                # ]]], dtype=np.int32)
                # cc_schedule_tables.append(cc_schedule_table)
                return cc_schedule_tables
            else:
                cc_schedule_table = np.array([[[
                    [0,  1,  0,  1],
                    [0,  1,  0,  1],
                    [2,  3,  2,  3],
                    [2,  3,  2,  3],
                ]]], dtype=np.int32)
        elif split_degrees[0] == 5:
            if causal:
                cc_schedule_table = np.array([[[
                    [ 0, -1, -1, -1, -1],
                    [ 0,  1, -1, -1, -1],
                    [ 1,  1,  2, -1, -1],
                    [ 0,  4,  2,  3, -1],
                    [ 3,  4,  2,  3,  4],
                ]]], dtype=np.int32)
            else:
                raise NotImplementedError()
        elif split_degrees[0] == 3:
            if causal:
                cc_schedule_table = np.array([[[
                    [ 0, -1, -1],
                    [ 0,  1, -1],
                    [ 0,  1,  2],
                ]]], dtype=np.int32)
            else:
                raise NotImplementedError()
        elif split_degrees[0] == 2:
            if causal:
                cc_schedule_table = np.array([[[
                    [ 0, -1],
                    [ 1,  1],
                ]]], dtype=np.int32)
            else:
                raise NotImplementedError()
        elif split_degrees[0] == 1:
            cc_schedule_table = np.array([[[
                [ 0 ],
            ]]], dtype=np.int32)
    return cc_schedule_table
