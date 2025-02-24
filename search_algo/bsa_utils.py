import numpy as np
from utils import Block_Type
from bsa_config import BSA_Repr, BSA_Config
from typing import List, Set

def split_to_node_configs(global_bsa_config: BSA_Config) -> Set[BSA_Config]:
    global_bsa_repr = global_bsa_config.bsa_repr
    node_bsa_reprs = global_bsa_repr.split_n(global_bsa_config.CP[1])   # after deduplcating
    CP0 = global_bsa_config.CP[0]
    node_bsa_configs = set([BSA_Config({'bsa_repr': r, 'CP': (CP0, 1)}) for r in node_bsa_reprs])
    return node_bsa_configs
        
def create_bsa_block_table(case_id: int) -> np.ndarray:
    if case_id == 0:
        # stride(1/16, 4, 3) (after remapping)
        block_table = np.full((16, 16), full_value=Block_Type.EMPTY, dtype=Block_Type)
        for r in range(4):
            for i in range(4):
                for j in range(4):
                    if (i, j) not in [(0, 3), (3, 0)]:
                        block_table[r * 4 + i, r * 4 + j] = Block_Type.FULL
    elif case_id == 1:
        # local+global(1/16, 1, 1) (no remapping)
        block_table = np.full((16, 16), full_value=Block_Type.EMPTY, dtype=Block_Type)
        for i in range(16):
            block_table[i, 0] = block_table[0, i] = block_table[i, i] = Block_Type.FULL
    else:
        raise Exception(f"Not support bsa_block_table case_id={case_id}")
    return block_table
    