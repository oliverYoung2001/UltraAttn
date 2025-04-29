import numpy as np
import inspect
from search_algo.utils import unique_list, Block_Comp_Volume
# from utils import Block_Type    # [ERROR]
# print(f'Block_Type in bsa_utils: {hex(id(Block_Type))}', flush=True)
# print(f'{inspect.getfile(Block_Type)}', flush=True)
from search_algo.utils import Block_Type
# print(f'Block_Type in bsa_utils: {hex(id(Block_Type))}', flush=True)
# print(f'{inspect.getfile(Block_Type)}', flush=True)

from search_algo.bsa_config import BSA_Repr, BSA_Config
from typing import List, Set, Optional

def bsa_is_dense(bsa_config: Optional[BSA_Config]):
    if bsa_config is None:
        return True
    if bsa_config.bsa_repr.block_table_raw.shape == (1, 1) and \
        bsa_config.bsa_repr.block_table_raw[0, 0].value in [Block_Type.FULL.value, Block_Type.CAUSAL.value]:
        return True
    return False

def bsa_is_causal(bsa_config: Optional[BSA_Config]):
    if bsa_config is None:
        return True
    if bsa_config.bsa_repr.block_table_raw.shape == (1, 1) and \
        bsa_config.bsa_repr.block_table_raw[0, 0].value == Block_Type.CAUSAL.value:
        return True
    return False

def convert_shape_config_to_str(shape_config: dict):
    return f"S={shape_config['S']}_Nh={shape_config['Nh']}_bs={shape_config['bs']}_D={shape_config['D']}"
    
# def get_bsa_comp_key(fob: int, CP: tuple, shape_config: dict, bsa_config: BSA_Config, key_suffix: str = ''):    # [DEPRECATED]
#     key_preffix = f'fob={fob}_CP={CP}_shape_config={{{convert_shape_config_to_str(shape_config)}}}_bsa_config={{{bsa_config}}}'
#     return f'{key_preffix}{key_suffix}'

def split_to_node_configs(global_bsa_config: BSA_Config) -> List[BSA_Config]:
    global_bsa_repr = global_bsa_config.bsa_repr
    node_bsa_reprs = global_bsa_repr.split_n(global_bsa_config.CP[1])   # after deduplcating
    CP0 = global_bsa_config.CP[0]   # intra CP
    node_bsa_configs = unique_list([BSA_Config(None, None, {'bsa_repr': r, 'CP': (CP0, 1)}) for r in node_bsa_reprs])
    return node_bsa_configs
        
def create_bsa_block_table(case_id: int) -> np.ndarray:
    if case_id == 0:
        # stride(1/16, 4, 3) (after remapping)
        block_table = np.full((16, 16), fill_value=Block_Type.EMPTY, dtype=Block_Type)
        for r in range(4):
            for i in range(4):
                for j in range(4):
                    if (i, j) not in [(0, 3), (3, 0)]:
                        block_table[r * 4 + i, r * 4 + j] = Block_Type.FULL
    elif case_id == 1:
        # local+global(1/16, 1, 1) (no remapping)
        block_table = np.full((16, 16), fill_value=Block_Type.EMPTY, dtype=Block_Type)
        for i in range(16):
            block_table[i, 0] = block_table[0, i] = block_table[i, i] = Block_Type.FULL
    elif case_id == 2:
        # BSA_Repr for star(1/4) (no remapping)
        block_table = np.full((4, 4), fill_value=Block_Type.EMPTY, dtype=Block_Type)
        for i in range(4):
            block_table[i, 0] = Block_Type.FULL
            block_table[i, i] = Block_Type.CAUSAL
    elif case_id == 3:
        # BSA_Repr for stream(1/8, 3) (no remapping)
        num_local = 3
        block_table = np.full((8, 8), fill_value=Block_Type.EMPTY, dtype=Block_Type)
        for i in range(8):
            for j in range(max(0, i - num_local + 1), i):
                block_table[i, j] = Block_Type.FULL
            block_table[i, 0] = Block_Type.FULL
            block_table[i, i] = Block_Type.CAUSAL
    elif case_id == 4:  # Full
        block_table = np.full((1, 1), fill_value=Block_Type.FULL, dtype=Block_Type)
    elif case_id == 5:  # Causal
        block_table = np.full((1, 1), fill_value=Block_Type.CAUSAL, dtype=Block_Type)
    else:
        raise Exception(f"Not support bsa_block_table case_id={case_id}")
    return block_table
    