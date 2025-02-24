import numpy as np
from search_algo.utils import Block_Comp_Volume, Block_Type, Block_Attention_Config, closest_fraction
from typing import Union, Optional
import regex as re
from typing import List, Optional
from __future__ import annotations

class BSA_Repr():   # OK
    """
    BSA Representation only !!! (without CP-aware)
    """
    def __init__(self, block_table: np.ndarray, cmap: Optional[np.ndarray]):
        assert len(block_table.shape) == 2 and len(cmap.shape) == 1 and \
               cmap.shape[0] == block_table.shape[0] == block_table.shape[1]
        self.block_table = block_table
        self.cmap = cmap
        self.block_table_raw, self.cmap_raw = self.simplify(self.block_table, self.cmap)
    
    def merge_blocks(self, sub_table: np.ndarray) -> Optional[Block_Type]:  # OK
        sub_par_d = sub_table.shape[0]
        
        is_empty, is_full, is_causal = True, True, True
        for ij in range(pow(sub_par_d, 2)):
            i, j = ij // sub_par_d, ij % sub_par_d
            # 1 Identify empty
            if sub_table[i, j] != Block_Type.EMPTY:
                is_empty = False
            # 2 Identify full
            if sub_table[i, j] != Block_Type.FULL:
                is_full = False
            # 3 Identify causal
            causal_target = Block_Type.CAUSAL if i == j else (Block_Type.FULL if i > j else Block_Type.EMPTY)
            if sub_table[i, j] != causal_target:
                is_causal = False

        if is_empty:
            return Block_Type.EMPTY
        elif is_full:
            return Block_Type.FULL
        elif is_causal:
            return Block_Type.CAUSAL
        return None
    
    def simplify_by_2(self, block_table: np.ndarray, cmap: np.ndarray): # OK
        par_d = block_table.shape[0]
        if par_d % 2 != 0:
            return block_table, cmap, False
        new_par_d = par_d // 2
        new_block_table = np.empty_like(block_table, shape=(new_par_d, new_par_d))
        successed = True
        # simplify block_table
        for ij in range(new_par_d * new_par_d):
            i = ij // new_par_d
            j = ij % new_par_d
            merged_elem = self.merge_blocks(block_table[i*2: (i+1)*2, j*2: (j+1)*2])
            if merged_elem is None:
                successed = False
                break
            new_block_table[i, j] = merged_elem
        
        if not successed:
            return block_table, cmap, successed
        # simplify cmap
        if cmap is not None:
            new_cmap = np.empty_like(cmap, (new_par_d))
            for i in range(new_par_d):
                if cmap[i * 2] != cmap[i * 2 + 1]:
                    new_cmap = None
                    break
                new_cmap[i] = cmap[i * 2]
        else:
            new_cmap = None
        return new_block_table, new_cmap, successed
    
    def simplify(self, block_table: np.ndarray, cmap: np.ndarray):  # OK
        while True:
            block_table, cmap, successed = self.simplify_by_2(block_table, cmap)
            if not successed:
                break
        return block_table, cmap

    def complicate_block(self, sub_table: np.ndarray, block_type: Block_Type, rate: int) -> np.ndarray:    # OK
        # sub_table = np.zeros((rate, rate), dtype=Block_Type)
        if block_type == Block_Type.EMPTY:
            for i in range(rate):
                for j in range(rate):
                    sub_table[i, j] = Block_Type.EMPTY
        elif block_type == Block_Type.FULL:
            for i in range(rate):
                for j in range(rate):
                    sub_table[i, j] = Block_Type.FULL
        elif block_type == Block_Type.CAUSAL:
            for i in range(rate):
                for j in range(rate):
                    causal_target = Block_Type.CAUSAL if i == j else (Block_Type.FULL if i > j else Block_Type.EMPTY)
                    sub_table[i, j] = causal_target
        else:
            raise Exception(f'[ERROR]: Unknown block_type: {block_type}')
        # return sub_table
    
    def complicate(self, block_table: np.ndarray, cmap: np.ndarray, rate: int): # OK
        par_d = block_table.shape[0]
        new_par_d = par_d * rate
        new_block_table = np.empty_like(block_table, shape=(new_par_d, new_par_d))
        # complicate block_table
        for i in range(par_d):
            for j in range(par_d):
                # [WARN]: check correctness
                self.complicate_block(new_block_table[i*rate: (i+1)*rate, j*rate, \
                                                     (j+1)*rate], block_table[i, j], rate)
        
        # complicate cmap
        if cmap is None:
            new_cmap = None
        else:
            new_cmap = np.empty_like(cmap, shape=(new_par_d))
            for i in range(par_d):
                new_cmap[i*rate: (i+1)*rate] = cmap[i]
        return new_block_table, new_cmap
    
    def split_n(self, n: int) -> List[BSA_Repr]:
        # [TODO]
        cur_spt = self.block_table_raw.shape[0]
        sub_bsa_reprs = []
        if cur_spt >= n:
            assert cur_spt % n == 0
            sub_size = cur_spt // n
            for i in range(n):
                for j in range(n):
                    sub_bsa_reprs.append(BSA_Repr(self.block_table_raw[i*sub_size: (i+1)*sub_size, j*sub_size: (j+1)*sub_size], None))
        else:
            assert n % cur_spt == 0
            sub_block_table, _ = self.complicate(self.block_table_raw, self.cmap_raw, rate=n // cur_spt)
            for i in range(n):
                for j in range(n):
                    sub_bsa_reprs.append(BSA_Repr(self.block_table_raw[i, j], None))
        # Deduplicate
        sub_bsa_reprs = list(set(sub_bsa_reprs))
        return sub_bsa_reprs
    
    def __eq__(self, other: BSA_Repr):
        return np.array_equal(self.block_table_raw, other.block_table_raw)
    
    def fingerprint(self):
        pass
    
class BSA_Config(): # OK
    """squ
    BSA Representation + CP-aware
    """
    def __init__(self, pat_dict: Union[dict, None] = None, pat_s: Optional[str] = None, pat_bsa_repr: Optional[dict] = None):# OK
        self.pat_dict = None
        self.pat_s = None
        self.block_table = None
        self.cmap = None
        self.bsa_repr = None
        if pat_s is not None and pat_dict is None:  # Create from pat_s
            pat_dict = self.convert_string_to_dict(pat_s)
        if pat_dict is not None:    # Create from pat_dict
            self.CP, self.Par_D, self.pattern_type, self.pattern_sparsity, \
                self.local_blocks, self.global_blocks, self.replicate = \
                pat_dict['CP'], pat_dict['Par_D'], pat_dict['pattern_type'], pat_dict['pattern_sparsity'], \
                pat_dict['local_blocks'], pat_dict['global_blocks'], pat_dict['replicate']
            self.pat_dict = pat_dict
            self.pat_s = self.convert_dict_to_string(self.pat_dict)
            if pat_s is not None:
                assert self.pat_s == pat_s
            # Sanity check begin
            assert self.pattern_type in ['lg', 'stride', 'star', 'stream'], f'[ERROR]: Unsupport pattern_type: {self.pattern_type}'
            # Sanity check end
            self.create_block_sparse_pattern()
            self.bsa_repr = BSA_Repr(self.block_table, self.cmap)
        elif pat_bsa_repr is not None:   # Create from BSA_Repr
            # assert ['block_table', 'cmap', 'CP'] <= list(pat_bsa_repr.keys())
            assert ['bsa_repr', 'CP'] <= list(pat_bsa_repr.keys())
            self.bsa_repr = pat_bsa_repr['bsa_repr']
            self.block_table = self.bsa_repr.block_table
            self.cmap = self.bsa_repr.cmap
            self.CP = pat_bsa_repr['CP']
        else:
            raise Exception(f'[ERROR]: Unknown BSA_Config __init__ !!!')
        # self.print_block_table()
        
    @classmethod
    def from_dict(cls, pat_dict: dict): # OK
        # CP: int, Par_D: int, pattern_type: str, pattern_sparsity: float, 
        #          local_blocks: tuple, global_blocks: tuple, replicate: int
        # dict: {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2},
        assert pat_dict is not None
        return cls(pat_dict)
    
    @classmethod
    def from_string(cls, pat_s: str): # OK
        assert pat_s is not None
        return cls(None, pat_s)

    def convert_dict_to_string(self, pat_dict: Union[dict, None]) -> Union[str, None]:  # OK
        if pat_dict is None:
            return None
        pattern_sparsity = closest_fraction(self.pattern_sparsity)
        numerator = pattern_sparsity.numerator
        denominator = pattern_sparsity.denominator
        return f'{self.CP}_{self.Par_D}_{self.pattern_type}_{numerator}-{denominator}_' \
               f'{self.local_blocks[0]}&{self.local_blocks[1]}_{self.global_blocks[0]}&{self.global_blocks[1]}_' \
               f'{self.replicate}'
               
    
    def convert_string_to_dict(self, pat_s: Optional[str]) -> Optional[dict]:   # OK
        if pat_s is None:
            return None
        D = r"\d+"
        S = r"[A-Za-z]+"
        re_pat_s = fr"\(({D}),({D})\)_({D})_({S})_({D})-({D})_({D})&({D})_({D})&({D})_({D})"
        match = re.match(re_pat_s, pat_s)
        if match:
            CP_intra, CP_inter, Par_D, pattern_type, ps_n, ps_d, lb_l, lb_r, gb_l, gb_r, r = match.groups()
            self.CP = (int(CP_intra), int(CP_inter))
            self.Par_D = int(Par_D)
            self.pattern_type = pattern_type
            self.pattern_sparsity = ps_n / ps_d
            self.local_blocks = (int(lb_l), int(lb_r))
            self.global_blocks = (int(gb_l), int(gb_r))
            self.replicate = int(r)
            pat_dict = {'CP': self.CP, 'Par_D': self.Par_D, 'pattern_type': self.pattern_type, 
                'pattern_sparsity': self.pattern_sparsity, 'local_blocks': self.local_blocks, 
                'global_blocks': self.global_blocks, 'replicate': self.replicate}
        else:
            raise Exception(f'[ERROR]: Regex not match !!!')
            pat_dict = None
        return pat_dict
    
    @property
    def total_sparsity(self):   # OK
        blk_num = 0
        for i in range(self.Par_D):
            for j in range(self.Par_D):
                blk_num += Block_Comp_Volume[self.block_table[i, j]]
        blk_sparsity = blk_num / pow(self.Par_D, 2)
        return blk_sparsity
                
    # def to_dict(self):
    #     return self.convert_dict_to_string(self.pat_dict)
    #     # return {'CP': self.CP, 'Par_D': self.Par_D, 'pattern_type': self.pattern_type, 
    #     #         'pattern_sparsity': self.pattern_sparsity, 'local_blocks': self.local_blocks, 
    #     #         'global_blocks': self.global_blocks, 'replicate': self.replicate}
        
    def to_string(self):    # OK
        return self.convert_dict_to_string(self.pat_dict)
               
    def __str__(self):  # OK
        return self.to_string()
        
    def generate_workload_table(self, ):  # 0 for empty, 1 for full, 2 for causal
        pass
    
    def generate_workload_partition_table(self, ):  # number stands for GPU rank it allocated
        pass
    
    def print_block_table(self):    # OK
        block_table_value = np.array([v.value for v in self.block_table.flatten()]).reshape(self.block_table.shape)
        print(f'block_table_value:\n{block_table_value}')
        
    def create_block_sparse_pattern_from_dict(self, pat_dict: dict) -> tuple:   # OK
        assert pat_dict is not None
        # if self.created:
        #     return
        # self.created = True
        # CP: int, ParD: int, pattern_type: str, pattern_sparsity: float, 
        #                         local_blocks = None, global_blocks = None, replicate: int = 1) -> Block_Attention_Config
        # dict: {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2},
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
        return block_table, cmap
        # block_config = Block_Attention_Config(CP, ParD, cmap, block_table)
        # return block_config
    
    def __eq__(self, other: BSA_Config):
        if self.CP != other.CP:
            return False
        return self.bsa_repr == other.bsa_repr
