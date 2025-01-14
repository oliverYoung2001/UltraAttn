import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, \
                                      get_profile_data, get_init_schedule_list, get_cc_optimal_schedule
from search_algo.dependent_graph import Dependent_Graph
from search_algo.graph_transformation_engine import Graph_Transformation_Engine
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from search_algo.bsa_config import BSA_Config

def get_exp_configs():
    # plan_type = 'automatic'
    plan_type = 'manual'  # for noncausal !!!
    # plan_type = 'ablation1'
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
    # da_config: SP=(1,8),Sg=(1024,1024),S=(8192,8192),Nh=(1,1),bs=1,D=128,causal=True,hierarchy=None

    hierarchy = None    # define in exps !!!
    
    # for Intra:
    SP0, SP1 = 1, 1
    SP0, SP1 = 1, 2
    SP0, SP1 = 1, 4
    SP0, SP1 = 1, 8
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

    Ss = [
        # 256, 
        # 512, 
        # 1 * 1024, 
        # 2 * 1024, 
        # 4 * 1024, 
        # 8 * 1024, 
        # 16 * 1024, 
        # 32 * 1024,    # fused failed on 8 * 8 !!!
        64 * 1024,    # fused failed on 4 * 8 !!!
    ]    # S per GPU
    # Ss = [16 * 1024]    # S per GPU

    # Nhq = Ng = 32
    # # Nhq = Ng = 1
    Nhs = [
        # 1,
        32,
    ]
    bs = 1
    D = 128
    causal = False
    causal = True
    use_BSA = False
    use_BSA = True  # prior than 'causal', this will hide 'causal'
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks(l&r), global_blocks(l&r), (replicate)
    # 8_8_lg_1-8_0_0&1
    # 8_8_lg_1-4_3_0_2
    BSA_patterns = [    # a dict
        # # stride_16_4_3 (8x2)
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 2},
        # stride_16_4_3 (8x4)
        {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (3, 3), 'global_blocks': (0, 0), 'replicate': 1},
        # # stride_16_4_3 (8x8)
        # # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 2), 'global_blocks': (0, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 1), 'global_blocks': (0, 0), 'replicate': 1},
        # # lg_16_1_1 (8x2)
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/8, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
        # # lg_16_1_1 (8x4)
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/4, 'local_blocks': (1, 1), 'global_blocks': (1, 1), 'replicate': 1},
        # # lg_16_1_1 (8x8)
        # # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (2, 2), 'global_blocks': (0, 0), 'replicate': 1},  # full
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (0, 1), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (0, 0), 'global_blocks': (1, 0), 'replicate': 1},
        # {'CP': 8, 'Par_D': 8, 'pattern_type': 'lg', 'pattern_sparsity': 1/2, 'local_blocks': (1, 1), 'global_blocks': (0, 0), 'replicate': 1},
    ]
    if use_BSA:
        assert SP1 == 8, f'[ERROR]: Now not support for degree of CP({SP1}) intra-node not equal to 8'
        BSA_configs = [BSA_Config(**p) for p in BSA_patterns]
    tot_sp = SP0 * SP1
    da_configs = []
    for Nh in Nhs:
        Nhq = Ng = Nh
        for S in Ss:
            Sq = Skv = S
            if use_BSA:
                for bsa_config in BSA_configs:
                    da_configs.append(Dist_Attn_Config((SP0, SP1), (Sq * tot_sp, Skv * tot_sp), (Nhq, Ng), bs, D, causal, hierarchy, bsa_config))
            else:
                da_configs.append(Dist_Attn_Config((SP0, SP1), (Sq * tot_sp, Skv * tot_sp), (Nhq, Ng), bs, D, causal, hierarchy))

    return da_configs

def get_exp_configs_for_search_workload_table():
    plan_type = 'automatic'
    # plan_type = 'manual'  # for noncausal !!!
    # plan_type = 'ablation1'
    MAX_QUEUE_SIZE = 100
    fobs = [
        # 0,
        1,
    ]
    # hierarchy = 0  # 0: intra-machine, 1: inter-machine
    hierarchy = None    # define in exps !!!
    transform_mode = 'bf'       # Enumerate all possible transformations
    transform_mode = 'greedy'   # Apply transformations greedily
    exp_configs = []
    for fob in fobs:
        exp_configs.append(Evaluation_Configs(plan_type, MAX_QUEUE_SIZE, fob, hierarchy=hierarchy, transform_mode=transform_mode))
    return exp_configs

def get_configs_for_search_workload_table():
    hierarchy = None
    
    # for Intra:
    SP0, SP1 = 1, 1
    # SP0, SP1 = 1, 2
    SP0, SP1 = 1, 4
    # SP0, SP1 = 1, 8
    # Sq = Skv = 16 * 1024   # 16k
    # Sq = Skv = 8 * 1024   # 8k
    
    
    # for Inter:
    # SP0, SP1 = 1, 8
    # SP0, SP1 = 2, 8
    # # SP0, SP1 = 3, 8
    # SP0, SP1 = 4, 8
    # # # SP0, SP1 = 6, 8
    # # SP0, SP1 = 8, 8
    # # SP0, SP1 = 16, 8

    Sq = Skv = 256   # S per GPU
    Sq = Skv = 512   # S per GPU
    # Sq = Skv = 1 * 1024   # S per GPU
    # Sq = Skv = 2 * 1024   # S per GPU
    # Sq = Skv = 4 * 1024   # S per GPU
    Sq = Skv = 8 * 1024   # S per GPU
    Ss = [
        # 256, 
        # 512, 
        1 * 1024, 
        # 2 * 1024, 
        # 4 * 1024, 
        # 8 * 1024, 
        # 16 * 1024, 
        # 32 * 1024,    # fused failed on 8 * 8!!!
        # 64 * 1024,    # fused failed !!!
    ]    # S per GPU
    # Ss = [16 * 1024]    # S per GPU

    # Nhq = Ng = 32
    # # Nhq = Ng = 1
    Nhs = [
        # 1,
        32,
    ]
    bs = 1
    D = 128
    causal = False
    causal = True
    
    tot_sp = SP0 * SP1
    da_configs = []
    for Nh in Nhs:
        Nhq = Ng = Nh
        for S in Ss:
            Sq = Skv = S
            da_configs.append(Dist_Attn_Config((SP0, SP1), (Sq * tot_sp, Skv * tot_sp), (Nhq, Ng), bs, D, causal, hierarchy))
    return da_configs

def show_schedule(schedule: Dist_Attn_Schedule, fob, name):
    print(f'{name}, schedule:\n{schedule.schedule_table}', flush=True)
    print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')
    if schedule.split_degrees[0] != schedule.da_config.SP[1] or schedule.split_degrees[1] != schedule.da_config.SP[1]:
        return
    
def run_cc_optimal_exp(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config):
    fob = exp_config.fob
    hierarchy = da_config.hierarchy
    print(f'plan_name: {da_config.get_plan_name(fob=0)}', flush=True)
    m_config = get_profile_data(da_config.SP)
    
    # for debugging Parallel Graph Transformation Engine
    schedules = get_cc_optimal_schedule(da_config, m_config)
    if isinstance(schedules, Dist_Attn_Schedule):
        schedules = [schedules]
    for schedule in schedules:
        print(f'schedule:\n{schedule.schedule_table}', flush=True)
        # print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')
        # print(f'fob: {fob}, get_relative_cc_time:{schedule.get_relative_cc_time()[fob]}')
        print(f'fob: {fob}, get_tot_comm_units: {schedule.get_tot_comm_units()[fob][0]}')
        schedule.get_relative_cc_time()
        balanced_r_cc_time = schedule.balanced_r_cc_time[fob]
        r_cc_time = schedule.r_cc_time[fob]
        print(f'get_relative_cc_time:\n{r_cc_time}')
        print(f'get_balanced_relative_cc_time: {np.max(balanced_r_cc_time[1:])}, '
            f'{np.max(balanced_r_cc_time[0])}\n{balanced_r_cc_time}')
        print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, '
            f'get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')

        d_graph = Dependent_Graph(schedule, exp_config.fob)
        execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=exp_config.plan_type)
        execute_plan.print_lp_result()
        
        gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
        gt_engine.transform(d_graph)
    # return
    return

def search_workload_table(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config):
    fob = exp_config.fob
    hierarchy = da_config.hierarchy = 1
    print(f'plan_name: {da_config.get_plan_name(fob=fob)}', flush=True)
    m_config = get_profile_data(da_config.SP, hierarchy=hierarchy)
    
    # # for debugging Parallel Graph Transformation Engine
    # schedules = get_cc_optimal_schedule(da_config, m_config)
    # if isinstance(schedules, Dist_Attn_Schedule):
    #     schedules = [schedules]
    # for schedule in schedules:
    #     print(f'schedule:\n{schedule.schedule_table}', flush=True)
    #     # print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')
    #     # print(f'fob: {fob}, get_relative_cc_time:{schedule.get_relative_cc_time()[fob]}')
    #     print(f'fob: {fob}, get_tot_comm_units: {schedule.get_tot_comm_units()[fob][0]}')
    #     schedule.get_relative_cc_time()
    #     balanced_r_cc_time = schedule.balanced_r_cc_time[fob]
    #     r_cc_time = schedule.r_cc_time[fob]
    #     print(f'get_relative_cc_time:\n{r_cc_time}')
    #     print(f'get_balanced_relative_cc_time: {np.max(balanced_r_cc_time[1:])}, '
    #         f'{np.max(balanced_r_cc_time[0])}\n{balanced_r_cc_time}')
    #     print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, '
    #         f'get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')

    #     d_graph = Dependent_Graph(schedule, exp_config.fob)
    #     execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=exp_config.plan_type)
    #     execute_plan.print_lp_result()
        
    #     gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
    #     gt_engine.transform(d_graph)
    # # return
    # return
    
    # Step1: Searching comp workloads scheduling
    init_schedule_list = get_init_schedule_list(da_config, m_config)
    search_engine = Search_Engine(exp_config, da_config, m_config, init_schedule_list)
    search_engine.search_optimal_schedules()
    return
    SCHEDULE_UNIQUE_ID = get_global_var('SCHEDULE_UNIQUE_ID')
    # print(f'tot schedules: {SCHEDULE_UNIQUE_ID}')
    # fwd
    fob = exp_config.fob
    print(f'[INFO]: fwd schedules')
    print(f'schedule_squeue_num[fwd]: {len(search_engine.schedule_queues[fob])}')
    par_dir = f'{os.path.dirname(__file__)}/execution_plans/SP{da_config.SP}_S{da_config.S}'
    if exp_config.plan_type == 'ablation1':
        par_dir = f'{par_dir}_{exp_config.plan_type}'
    os.makedirs(par_dir, exist_ok=True)
    # return
    # for schedule in init_schedule_list:
    #     show_schedule(schedule, fob, 'init')

    # Step2: DAG Transformation
    # Step3: Execution Plan Generation
    QUEUE_LEN = len(search_engine.schedule_queues[fob])
    for _ in range(len(search_engine.schedule_queues[fob])):
        schedule = search_engine.schedule_queues[fob].pop()
        show_schedule(schedule, fob, f'alg{_}')
        d_graph = Dependent_Graph(schedule, fob, 1) # Intra-machine
        execute_plan = Execution_Plan(d_graph, fob, exp_config.plan_type)
        execute_plan.print_lp_result()
        # if _ == 1:  # qo schedule
        #     example_schedule = schedule
        # if _ == 2:  # kv schedule
        #     example_schedule = schedule
        # if _ == 3:  # real example
        #     example_schedule = schedule
        # dump plan
        plan_name = execute_plan.get_plan_name()
        plan_file = f'{par_dir}/{plan_name}_alg{_}.pkl'
        print(f'plan_file: {plan_file}')
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
    return
    d_graph = Dependent_Graph(example_schedule, fob, 1) # Intra-machine
    execute_plan = Execution_Plan(d_graph, fob)
    # dump plan
    plan_name = execute_plan.get_plan_name()
    plan_file = f'{os.path.dirname(__file__)}/execution_plans/{plan_name}.pkl'
    with open(plan_file, 'wb') as f:
        pickle.dump(execute_plan, f)
    # load plan
    with open(plan_file, 'rb') as f:
        execute_plan_loaded = pickle.load(f)
    execute_plan_loaded.print_lp_result()

def generate_inter_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config):
    hierarchy = exp_config.hierarchy = da_config.hierarchy = 0
    m_config = get_profile_data(da_config.SP, exp_config.hierarchy)
    cc_optimal_schedule = get_cc_optimal_schedule(da_config, m_config)
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print(f'cc_optimal_schedule: {cc_optimal_schedule.schedule_table}')
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
    par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/inter_SP{da_config.hierarchy_sp}_fob={exp_config.fob}'
    os.makedirs(par_dir, exist_ok=True)
    plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    for plan_type in plan_types:
        # Raw Execution_Plan:
        print(f'Raw, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True) # use 'level0'
        # if not plan_type == 'automatic':
        if True:
            cur_level = 0
            cc_optimal_schedule.m_config.comp_profile_maps[hierarchy].change_current_level(cur_level)  # [NOTE]: Use it before Graph construction
            d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type)
            execute_plan.print_lp_result()
            plan_name = execute_plan.get_plan_name()
            if plan_type == 'ablation1':
                plan_name = f'{plan_name}_{plan_type}'
            # Dump Execution_Plan:
            plan_file = f'{par_dir}/{plan_name}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        
        # Transformed Execution_Plans:
        print(f'Fused, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True)   # use 'level1' and 'level2'
        cur_level = 2 if plan_type == 'automatic' else 1
        cc_optimal_schedule.m_config.comp_profile_maps[hierarchy].change_current_level(cur_level)  # [NOTE]: Use it before Graph construction
        d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
        gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
        execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=plan_type)
        if execute_plan is None:    # No feasible transformations
            continue
        assert isinstance(execute_plan, Execution_Plan)
        plan_name = f'{execute_plan.get_plan_name()}_fused'
        if plan_type == 'ablation1':
            plan_name = f'{plan_name}_{plan_type}'
        # Dump Execution_Plan:
        plan_file = f'{par_dir}/{plan_name}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)

def generate_intra_execution_plans(exp_config: Evaluation_Configs, da_config: Dist_Attn_Config):
    exp_config.hierarchy = da_config.hierarchy = 1
    m_config = get_profile_data(da_config.SP, exp_config.hierarchy)
    cc_optimal_schedule = get_cc_optimal_schedule(da_config, m_config)
    if not isinstance(cc_optimal_schedule, Dist_Attn_Schedule):
        assert isinstance(cc_optimal_schedule, list)
        cc_optimal_schedule = cc_optimal_schedule[0]
    print(f'cc_optimal_schedule.schedule_table: \n{cc_optimal_schedule.schedule_table}')
    d_graph = Dependent_Graph(cc_optimal_schedule, exp_config.fob)
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)

    if da_config.bsa_config:
        print(f'bsa_config_string: {da_config.bsa_config.to_string()}')
        par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{da_config.hierarchy_sp}_fob={exp_config.fob}_{da_config.bsa_config}'
    else:
        par_dir = f'{os.path.dirname(__file__)}/execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP{da_config.hierarchy_sp}_fob={exp_config.fob}_causal={da_config.causal}'
    print(f'par_dir: {par_dir}')
    os.makedirs(par_dir, exist_ok=True)
    # return
    plan_types = ['automatic', 'ablation1'] # ILP, Flexflow
    for plan_type in plan_types:
        # Raw Execution_Plan:
        print(f'Raw, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True)
        # if not plan_type == 'automatic':
        if True:
            execute_plan = Execution_Plan(d_graph, exp_config.fob, plan_type=plan_type)
            execute_plan.print_lp_result()
            plan_name = execute_plan.get_plan_name()
            if plan_type == 'ablation1':
                plan_name = f'{plan_name}_{plan_type}'
            # Dump Execution_Plan:
            plan_file = f'{par_dir}/{plan_name}.pkl'
            with open(plan_file, 'wb') as f:
                pickle.dump(execute_plan, f)
        
        # Transformed Execution_Plans:
        print(f'Fused, {"ILP" if plan_type == "automatic" else "Flexflow"}:', flush=True)
        gt_engine = Graph_Transformation_Engine(exp_config, da_config, m_config)
        execute_plan = gt_engine.transform(d_graph, exp_config.transform_mode, plan_type=plan_type)
        if execute_plan is None:    # No feasible transformations
            continue
        assert isinstance(execute_plan, Execution_Plan)
        plan_name = f'{execute_plan.get_plan_name()}_fused'
        if plan_type == 'ablation1':
            plan_name = f'{plan_name}_{plan_type}'
        # Dump Execution_Plan:
        plan_file = f'{par_dir}/{plan_name}.pkl'
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
 
def main():
    # [Deprecated]: begin. Use ILP to workload partition !!!
    # da_configs = get_configs_for_search_workload_table()
    # exp_configs = get_exp_configs_for_search_workload_table()
    # for da_config in da_configs:
    #     for exp_config in exp_configs:
    #         search_workload_table(exp_config, da_config)
    # [Deprecated]: end. 
    
    # return
    da_configs = get_configs()
    exp_configs = get_exp_configs()
    if isinstance(da_configs, Dist_Attn_Config):
        da_configs = [da_configs]
    if isinstance(exp_configs, Evaluation_Configs):
        exp_configs = [exp_configs]
    for exp_config in exp_configs:
        print(f'exp_config: {exp_config}', flush=True)
        for da_config in da_configs:
            print(f'da_config: {da_config}', flush=True)
            # run_cc_optimal_exp(exp_config, da_config)
            # run_exp(exp_config, da_config)
            # generate_inter_execution_plans(exp_config, da_config) # for causal
            generate_intra_execution_plans(exp_config, da_config)   # for causal
            # generate_intra_execution_plans_for_BSA(exp_config, da_config)

if __name__ == '__main__':
    main()