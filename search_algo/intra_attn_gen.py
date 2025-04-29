import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Machine_Config, Dist_Attn_Config, \
    get_profile_data, get_init_schedule_list, create_schedule
from search_algo.dependent_graph import Dependent_Graph
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from functools import partial
from search_algo.initialize import initialize_prof_db
from search_algo.dense_utils import create_plan_for_full, write_plan

def get_configs():
    SP0, SP1 = 1, 1
    Sq = Skv = 1 * 1024   # 2k
    SP0, SP1 = 1, 2
    Sq = Skv = 2 * 1024   # 2k
    SP0, SP1 = 1, 4
    Sq = Skv = 4 * 1024   # 4k
    SP0, SP1 = 1, 8
    Sq = Skv = 16 * 1024   # 16k
    # Sq = Skv = 8 * 1024   # 8k
    
    Nhq = Ng = 32
    bs = 1
    D = 128
    causal = False
    # causal = True
    hierarchy = 1
    return Dist_Attn_Config((SP0, SP1), (Sq, Skv), (Nhq, Ng), bs, D, causal, hierarchy)


def main():
    CLUSTER_NAME = os.environ.get('CLUSTER_NAME', None)
    PLATFORM = os.environ.get(f'PLATFORM', None)
    assert CLUSTER_NAME in ['qiyuan', 'fit', 'hamming'], f'[ERROR]: Not support CLUSTER_NAME: {CLUSTER_NAME}'
    assert PLATFORM in ['A100', 'A800', 'H800', 'H100'], f'[ERROR]: Not support PLATFORM: {PLATFORM}'
    fobs = [
        0,
        1,
    ]
    da_config = get_configs()
    # Initialize Profile_DataBase
    prof_db = initialize_prof_db()
    
    # m_config = get_profile_data(da_config.SP, da_config.hierarchy)
    tot_sp = da_config.SP[0] * da_config.SP[1]
    for fob in fobs:
        par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{da_config.SP[1]}_fob={fob}'
        os.makedirs(par_dir, exist_ok=True)
        for X in range(1, tot_sp + 1):
            if tot_sp % X != 0:
                continue
            if X == 1 or X == tot_sp:
                plan = create_plan_for_full(da_config, prof_db.m_config, X, fob=fob, first_dim=0)
                write_plan(plan, prefix=par_dir)
            else:
                for first_dim in range(1):  # [TODO]: Support first_dim == 1
                    plan = create_plan_for_full(da_config, prof_db.m_config, X, fob=fob, first_dim=first_dim)
                    write_plan(plan, prefix=par_dir)
    
if __name__ == '__main__':
    main()