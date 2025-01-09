from utils import Block_Comp_Volume, Block_Type, Block_Attention_Config
import numpy as np
# from typing import Unions

def create_block_sparse_pattern(CP: int, ParD: int, pattern_type: str, pattern_sparsity: float, 
                                local_blocks = None, global_blocks = None, replicate: int = 1) -> Block_Attention_Config:
    # For example: 
    # pattern_type: "stream"  # causal, full, star, stream, stride
    # pattern_sparsity: 0.125 # 1/8
    # local_blocks: 3 # 3 local blocks
    chunk_num = int(round(1 / pattern_sparsity))
    assert ParD % CP == 0, "ParD must be divisible by CP"
    assert ParD % (chunk_num * replicate) == 0, "ParD must be divisible by chunk_num"
    cmap = np.array([i // (ParD // CP) for i in range(ParD)]) # (0, 0, 1, 1, ..., CP-1, CP-1)
    block_table = np.zeros((ParD, ParD), dtype=Block_Type)
    # clear block_table to empty
    for i in range(ParD):
        for j in range(ParD):
            block_table[i, j] = Block_Type.EMPTY
    if pattern_type in ['star', 'stream']:  # attention_sink_and_local_pattern (causal)
        if pattern_type == 'star':
            assert local_blocks is None or local_blocks == 1, "local_blocks must be 1 for star pattern"
        if local_blocks is None:
            local_blocks = 1
        # sink part
        for i in range(ParD):
            for j in range(min(ParD // chunk_num * 1, i)):
                block_table[i, j] = Block_Type.FULL
        # local part
        for c_i in range(chunk_num):
            for c_j in range(max(0, c_i - local_blocks + 1), c_i + 1):
                for p_x in range(ParD // chunk_num * c_i, ParD // chunk_num * (c_i + 1)):
                    for p_y in range(ParD // chunk_num * c_j, ParD // chunk_num * (c_j + 1)):
                        if p_x < p_y:
                            continue
                        block_table[p_x, p_y] = Block_Type.CAUSAL if p_x == p_y else Block_Type.FULL
        # Example:
        # CP, ParD, chunk_num, local_blocks = 4, 8, 4, 1
        # [C,  ,  ,  ,  ,  ,  ,  ]
        # [F, C,  ,  ,  ,  ,  ,  ]
        # [F, F, C,  ,  ,  ,  ,  ]
        # [F, F, F, C,  ,  ,  ,  ]
        # [F, F,  ,  , C,  ,  ,  ]
        # [F, F,  ,  , F, C,  ,  ]
        # [F, F,  ,  ,  ,  , C,  ]
        # [F, F,  ,  ,  ,  , F, C]
        
    elif pattern_type in ['stride']:
        pass
    elif pattern_type in ['local_global']:  # contains stride_remap_pattern
        if isinstance(local_blocks, int):
            local_blocks = (local_blocks, local_blocks)
        assert len(local_blocks) == 2, "local_blocks must be a tuple of 2 elements"
        if isinstance(global_blocks, int):
            global_blocks = (global_blocks, global_blocks)
        assert len(global_blocks) == 2, "global_blocks must be a tuple of 2 elements"
        assert local_blocks is not None and global_blocks is not None, "local_blocks and global_blocks must be specified"
        for r in range(replicate):
            sub_ParD = ParD // replicate
            r_offset = sub_ParD * r
            # 1. global part
            # 1.1 row global part
            for i in range(global_blocks[0] * (sub_ParD // chunk_num)):
                for j in range(sub_ParD):
                    block_table[i + r_offset, j + r_offset] = Block_Type.FULL
            # 1.2 column global part
            for j in range(global_blocks[1] * (sub_ParD // chunk_num)):
                for i in range(sub_ParD):
                    block_table[i + r_offset, j + r_offset] = Block_Type.FULL
            # 2. local part
            for c_i in range(chunk_num):
                for c_j in range(max(0, c_i - local_blocks[0] + 1), min(chunk_num, c_i + local_blocks[1])):
                    for p_x in range(sub_ParD // chunk_num * c_i, sub_ParD // chunk_num * (c_i + 1)):
                        for p_y in range(sub_ParD // chunk_num * c_j, sub_ParD // chunk_num * (c_j + 1)):
                            block_table[p_x + r_offset, p_y + r_offset] = Block_Type.FULL
        # Example:
        # CP, ParD, local_blocks, global_blocks = 4, 8, (2, 2), 1
        # [F, F, F, F, F, F, F, F,]
        # [F, F, F, F, F, F, F, F]
        # [F, F, F, F, F, F,  ,  ]
        # [F, F, F, F, F, F,  ,  ]
        # [F, F, F, F, F, F, F, F]
        # [F, F, F, F, F, F, F, F]
        # [F, F,  ,  , F, F, F, F]
        # [F, F,  ,  , F, F, F, F]
    else:
        raise ValueError(f"Invalid pattern type: {pattern_type}")
    block_config = Block_Attention_Config(CP, ParD, cmap, block_table)
    return block_config
        