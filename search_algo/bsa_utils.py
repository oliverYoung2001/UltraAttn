import numpy as np
import inspect
from search_algo.utils import unique_list, Block_Comp_Volume
from search_algo.utils import Block_Type
from search_algo.bsa_config import BSA_Repr, BSA_Config
from typing import List, Set, Optional, Union
import regex as re
from search_algo.database import Prof_DB

bsa2full_comp_key_suffixes_map_dict: dict = {}

def select_best_profile_comp_key(key_preffix, profile_comp_key_suffixes, comp_profile_map, fob: bool, fault_tolerance_func) -> str:
    opt_comp_time = float('inf')
    for key_suffix in profile_comp_key_suffixes:
        comp_map_key = f'{key_preffix}{key_suffix}'
        try:
            cur_comp_time = comp_profile_map.get_comp_time_from_map_key(comp_map_key)[fob]
        except Exception as e:
            if not fault_tolerance_func(key_suffix):
                # When Nh=32 and intra_bsa=full and inter_bsa is dense, 'w_kernel_tile' does not exist !!!
                raise e
        if cur_comp_time < opt_comp_time:
            selected_key_suffix = key_suffix
            opt_comp_time = cur_comp_time
    return f'{key_preffix}{selected_key_suffix}'
    
def convert_intra_profile_key_to_exe_key_or_plan(comp_map_key, prof_db: Prof_DB):
    # Objective: convert intra full profile key to inducted exe key or exe plan generated
    # For example: 
    # 1. 'fob=0_CP=(8, 1)_shape_config={S=(4096, 16384)_Nh=(1, 1)_bs=1_D=128}_bsa_config={CP=(8, 1)_repr=[[1]]}_ablation=(w/o_kernel_tile,Y=2,X=4,dim=0)'
    # to 'fob=0_CP=(8, 1)_shape_config=full_ablation=(Y=2,X=4,dim=0)'
    # 2. 'fob=0_CP=(8, 1)_shape_config={S=(4096, 16384)_Nh=(1, 1)_bs=1_D=128}_bsa_config={CP=(8, 1)_repr=[[1]]}_ablation=(w_kernel_tile,Y=4,X=2,dim=0)'
    # to Fused_Execution_Plan(Y, X, - 0.0, causal=False, fob=fob)
    # Return: exe_key or exe_plan(Fused_Execution_Plan)
    if 'Y=' in comp_map_key:    # intra full of inter causal
        if 'w/o_kernel_tile' in comp_map_key:
            origin_pat = r'shape_config=.*_ablation=\(w/o_kernel_tile,'
            replace_pat = r'shape_config=full_ablation=('
            exe_key = re.sub(origin_pat, replace_pat, comp_map_key)
            return exe_key
        elif 'w_kernel_tile' in comp_map_key:
            pat = r'^fob=(\d+)_.*Y=(\d+),X=(\d+),.*$'
            matched = re.match(pat, comp_map_key)
            assert matched is not None, f'Regex failed when deals with Fused_Execution_Plan'
            fob, Y, X = int(matched.group(1)), int(matched.group(2)), int(matched.group(3))
            from search_algo.execute_plan import Fused_Execution_Plan
            return Fused_Execution_Plan(Y, X, None, fob=fob, m_config=prof_db.m_config)
        else:
            assert False, f'Not supported comp_map_key={comp_map_key}'
    else:
        return comp_map_key
    
def get_b2f_suf_map(CP: int) -> dict:
    global bsa2full_comp_key_suffixes_map_dict
    # print(f'CP: {CP}', flush=True)
    if CP not in bsa2full_comp_key_suffixes_map_dict.keys():
        tmp_map = {}
        XYs = [(X, CP // X) for X in range(1, CP + 1) if CP % X == 0]
        # print(f'XYs: {XYs}')
        wo_k_sufs = [f'_ablation=(w/o_kernel_tile,Y={Y},X={X},dim=0)' for X, Y in XYs]
        w_k_sufs = [f'_ablation=(w_kernel_tile,Y={Y},X={X},dim=0)' for X, Y in XYs]
        tmp_map = {
            '_ablation=(w/o_gpu_tile,w/o_kernel_tile,Flexflow)': [f'_ablation=(w/o_kernel_tile,Y={CP},X={1},dim=0)'],
            '_ablation=(w/o_kernel_tile,Flexflow)': wo_k_sufs,
            '_ablation=(w_kernel_tile,Flexflow)':   wo_k_sufs + w_k_sufs,
            '_ablation=(w/o_kernel_tile,ILP)':      wo_k_sufs,
            '_ablation=(w_kernel_tile,ILP)':        wo_k_sufs + w_k_sufs,
        }
        bsa2full_comp_key_suffixes_map_dict[CP] = tmp_map
    return bsa2full_comp_key_suffixes_map_dict[CP]

def bsa_repr_is_square_full(bsa_repr: BSA_Repr):
    if bsa_repr.block_table_raw.shape == (1, 1) and \
        bsa_repr.block_table_raw[0, 0].value == Block_Type.FULL.value:
        return True
    return False

def bsa_is_square_full(bsa_config: BSA_Config):
    return bsa_repr_is_square_full(bsa_config.bsa_repr)

def bsa_is_full(bsa_config: BSA_Config):
    block_table = bsa_config.bsa_repr.block_table_raw
    for i in range(block_table.shape[0]):
        for j in range(block_table.shape[1]):
            if block_table[i, j].value != Block_Type.FULL.value:
                return False
    return True

def bsa_repr_is_causal(bsa_repr: BSA_Repr):
    if bsa_repr.block_table_raw.shape == (1, 1) and \
        bsa_repr.block_table_raw[0, 0].value == Block_Type.CAUSAL.value:
        return True
    return False

def bsa_is_causal(bsa_config: BSA_Config):
    return bsa_repr_is_causal(bsa_config.bsa_repr)

def bsa_is_dense(bsa_config: BSA_Config):
    return bsa_is_causal(bsa_config) or bsa_is_full(bsa_config)

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
    