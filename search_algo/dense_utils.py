import numpy as np
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Machine_Config, Dist_Attn_Config, create_schedule
import pickle
from functools import partial
from search_algo.dependent_graph import Dependent_Graph
from search_algo.execute_plan import Execution_Plan
from typing import List, Optional
from search_algo.bsa_config import BSA_Config
from search_algo.utils import Block_Type
import copy

def get_block_schedule_table_for_full(split_degrees: list, S_map: np.ndarray, causal: bool, X, da_config):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    assert split_degrees[0] == split_degrees[1] and split_degrees[0] % X == 0
    Y = split_degrees[0] // X
    block_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    block_schedule_table -= 1  # -1 means not used
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    block_schedule_table[i, j, k, l] = S_map[i, k // X * X + l % X]
    return block_schedule_table

def create_plan_for_full(da_config: Dist_Attn_Config, m_config: Machine_Config, X, fob, first_dim) -> Execution_Plan:
    # **Not fused** with manually cc schedule !!!
    tot_sp = da_config.tot_sp
    # Create Schedule:
    split_degrees = [tot_sp, tot_sp, 1, 1]
    S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
    S_map[:] = np.arange(tot_sp)
    get_schedule_table_func = partial(get_block_schedule_table_for_full, X=X)
    schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_schedule_table_func)
    print(f'schedule: {schedule.schedule_table}', flush=True)
    # Create Dependent Graph:
    d_graph = Dependent_Graph(schedule, fob) # Intra-machine
    # Create Execution Plan:
    plan = Execution_Plan(d_graph, fob, plan_type=None, is_hack=False)
    # Generate Manual Plan:
    plan.generate_manual_plan(tot_sp, X, first_dim=first_dim)
    return plan

def write_plan(execute_plan: Execution_Plan, prefix: str):
    # dump plan
    plan_name = execute_plan.get_plan_name()
    plan_file = f'{prefix}/{plan_name}.pkl'
    with open(plan_file, 'wb') as f:
        pickle.dump(execute_plan, f)
    # load plan
    with open(plan_file, 'rb') as f:
        execute_plan_loaded = pickle.load(f)
    execute_plan_loaded.print_lp_result()

def split_dense_configs(dense_configs: List[BSA_Config]):
    # split dense configs to full and causal
    full_configs, causal_configs = [], []
    for dense_config in dense_configs:
        assert dense_config.bsa_repr.block_table_raw.shape == (1, 1)
        if dense_config.bsa_repr.block_table_raw[0, 0].value == Block_Type.FULL.value:
            full_configs.append(dense_config)
        elif dense_config.bsa_repr.block_table_raw[0, 0].value == Block_Type.CAUSAL.value:
            causal_configs.append(dense_config)
        else:
            raise Exception(f'[ERROR]: Empty dense_config found !!!')
    return full_configs, causal_configs

def create_ablation_configs_for_full(hierarchy_cp, KERNEL_TILE_TYPEs: Optional[List] = None):
    ablation_dicts = []
    if KERNEL_TILE_TYPEs is None:
        KERNEL_TILE_TYPEs = ['w/o_kernel_tile']
    for KERNEL_TILE_TYPE in KERNEL_TILE_TYPEs:
        for X in range(1, hierarchy_cp + 1):
            if hierarchy_cp % X != 0:
                continue
            Y = hierarchy_cp // X
            if X == 1 or X == hierarchy_cp:
                first_dims = [0]
            else:
                first_dims = [0]    # [TODO]: Support first_dim == 1
            for first_dim in first_dims:
                ad = {
                    'Y': Y,
                    'X': X,
                    'first_dim': first_dim,
                    'KERNEL_TILE_TYPE': KERNEL_TILE_TYPE,
                }
                ablation_dicts.append(ad)
    return ablation_dicts
    
def create_plan_key_suffixes_for_full(hierarchy_cp):
    ablation_dicts = create_ablation_configs_for_full(hierarchy_cp)
    key_suffixes = [f"_ablation=(Y={ad['Y']},X={ad['X']},dim={ad['first_dim']})" for ad in ablation_dicts]
    return key_suffixes
