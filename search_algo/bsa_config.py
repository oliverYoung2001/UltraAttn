import numpy as np
from search_algo.utils import Block_Comp_Volume, Block_Type, Block_Attention_Config, closest_fraction

class BSA_Config():
    def __init__(self, CP: int, Par_D: int, pattern_type: str, pattern_sparsity: float, 
                 local_blocks: tuple, global_blocks: tuple, replicate: int):
        # Sanity check
        assert pattern_type in ['lg', 'stride', 'star', 'stream'], f'[ERROR]: Unsupport pattern_type: {pattern_type}'
        self.CP = CP
        self.Par_D = Par_D
        self.pattern_type = pattern_type
        self.pattern_sparsity = pattern_sparsity
        self.local_blocks = local_blocks
        self.global_blocks = global_blocks
        self.replicate = replicate
        self.block_table = None
        self.cmap = None
        self.create_block_sparse_pattern()
        # self.print_block_table()
    
    @property
    def total_sparsity(self):
        blk_num = 0
        for i in range(self.Par_D):
            for j in range(self.Par_D):
                blk_num += Block_Comp_Volume[self.block_table[i, j]]
        blk_sparsity = blk_num / pow(self.Par_D, 2)
        return blk_sparsity
                
    def to_dict(self):
        return {'CP': self.CP, 'Par_D': self.Par_D, 'pattern_type': self.pattern_type, 
                'pattern_sparsity': self.pattern_sparsity, 'local_blocks': self.local_blocks, 
                'global_blocks': self.global_blocks, 'replicate': self.replicate}

        
    def to_string(self):
        pattern_sparsity = closest_fraction(self.pattern_sparsity)
        numerator = pattern_sparsity.numerator
        denominator = pattern_sparsity.denominator
        return f'{self.CP}_{self.Par_D}_{self.pattern_type}_{numerator}-{denominator}_' \
               f'{self.local_blocks[0]}&{self.local_blocks[1]}_{self.global_blocks[0]}&{self.global_blocks[1]}_' \
               f'{self.replicate}'
               
    def __str__(self):
        return self.to_string()
        
    def generate_workload_table(self, ):  # 0 for empty, 1 for full, 2 for causal
        pass
    
    def generate_workload_partition_table(self, ):  # number stands for GPU rank it allocated
        pass
    
    def print_block_table(self):
        block_table_value = np.array([v.value for v in self.block_table.flatten()]).reshape(self.block_table.shape)
        print(f'block_table_value:\n{block_table_value}')
        
    def create_block_sparse_pattern(self):
        # if self.created:
        #     return
        # self.created = True
        # CP: int, ParD: int, pattern_type: str, pattern_sparsity: float, 
        #                         local_blocks = None, global_blocks = None, replicate: int = 1) -> Block_Attention_Config
        CP, ParD, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = \
            self.CP, self.Par_D, self.pattern_type, self.pattern_sparsity, \
            self.local_blocks, self.global_blocks, self.replicate
        # For example: 
        # pattern_type: "stream"  # causal, full, star, stream, stride
        # pattern_sparsity: 0.125 # 1/8
        # local_blocks: 3 # 3 local blocks
        chunk_num = int(round(1 / pattern_sparsity))
        assert ParD % CP == 0, "ParD must be divisible by CP"
        assert ParD % (chunk_num * replicate) == 0, "ParD must be divisible by chunk_num"
        # [TODO]: Explore the best cmap !!!
        cmap = np.array([i // (ParD // CP) for i in range(ParD)]) # Naive cmap, (0, 0, 1, 1, ..., CP-1, CP-1)
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
        elif pattern_type in ['lg']:  # contains stride_remap_pattern
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
        self.cmap = cmap
        self.block_table = block_table
        # block_config = Block_Attention_Config(CP, ParD, cmap, block_table)
        # return block_config
        