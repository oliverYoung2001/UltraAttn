import pulp
import time
import math
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from enum import Enum
from typing import Union
from utils import Block_Comp_Volume, Block_Type, Block_Attention_Config
from custom_sparse_pattern import create_block_sparse_pattern

TIME_BUDGET = 5 * 60 * 60   # 5 hours
    
# def print_lp_result_ILP(N: int, Vars: dict):
#     print(f'LP result:', flush=True)
#     for k, v in Vars.items():
#         print(f'{k}: {v.varValue}')
#     print(f'Allocation Matrix:')
#     for i in range(N):
#         for j in range(N):
#             if i < j:
#                 print(f'/', end=' ')
#                 continue
#             if i == j:
#                 print(f'{i}', end=' ')
#                 continue
#             for k in range(N):
#                 if Vars[f'x_{i}_{j}_{k}'].varValue == 1:
#                     print(f'{k}', end=' ')
#         print(f'')

def print_lp_result(CP: int, ParD: int, Vars: dict, v_attr: str, cmap: Union[None, np.ndarray] = None, diagonal_full = True):
    print(f'LP result:', flush=True)
    for k, v in Vars.items():
        assert hasattr(v, v_attr)
        print(f'{k}: {getattr(v, v_attr)}')
    print(f'Allocation Matrix:')
    for i in range(ParD):
        for j in range(ParD):
            if i == j and diagonal_full:
                print(f'{cmap[i]}{" " if cmap[i] < 10 else ""}', end=' ')
                continue
            if f'x_{i}_{j}_{0}' not in Vars.keys(): # [NOTE]: error for lagency code !!!
                print(f'/ ', end=' ')
                continue
            # if i == j:
            #     print(f'{i}{" " if i < 10 else ""}', end=' ')
            #     continue
            max_v = 0
            max_kid = -1
            for k in range(CP):
                cur_v = getattr(Vars[f'x_{i}_{j}_{k}'], v_attr)
                if cur_v > max_v:
                    max_v = cur_v
                    max_kid = k
            print(f'{max_kid}{" " if max_kid < 10 else ""}', end=' ')
        print(f'')

def ILP(N: int):
    # causal = True, fwd
    mylp = pulp.LpProblem(f"Workload_Partition_Allocation_ILP", pulp.LpMinimize)

    # Variables
    Var_cat_default = 'Continuous'
    Var_cat_default = 'Integer'
    Vars = dict()
    for i in range(N):
        for j in range(i):
            for k in range(N):
                Vars[f"x_{i}_{j}_{k}"] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat='Binary')
    for g in range(N):
        for i in range(N):
            Vars[f'a_{g}_{i}'] = pulp.LpVariable(f'a_{g}_{i}', cat=Var_cat_default, lowBound=0, upBound=1)
    for g in range(N):
        for j in range(N):
            Vars[f'b_{g}_{j}'] = pulp.LpVariable(f'b_{g}_{j}', cat=Var_cat_default, lowBound=0, upBound=1)
    for g in range(N):
        Vars[f'A_{g}'] = pulp.LpVariable(f'A_{g}', cat=Var_cat_default, lowBound=0, upBound=N-1)
        Vars[f'B_{g}'] = pulp.LpVariable(f'B_{g}', cat=Var_cat_default, lowBound=0, upBound=N-1)
        Vars[f'C_{g}'] = pulp.LpVariable(f'C_{g}', cat=Var_cat_default, lowBound=0, upBound=N-1)
        Vars[f'D_{g}'] = pulp.LpVariable(f'D_{g}', cat=Var_cat_default, lowBound=0, upBound=N-1)
    Vars[f'Comm_Volume'] = pulp.LpVariable(f'Comm_Volume', cat=Var_cat_default, lowBound=0)
    
    # Constraints
    # 0. Workload Partition
    for i in range(N):
        for j in range(i):
            mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) == 1 # w/o Replication
            # mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) >= 1 # w Replication
    # 1. a, b
    for g in range(N):
        for i in range(N):
            for j in range(i):
                mylp += Vars[f'a_{g}_{i}'] >= Vars[f'x_{i}_{j}_{g}']
    for g in range(N):
        for j in range(N):
            for i in range(j + 1, N):
                mylp += Vars[f'b_{g}_{j}'] >= Vars[f'x_{i}_{j}_{g}']
    
    # 2. A, B, C, D
    for g in range(N):
        mylp += Vars[f'A_{g}'] == pulp.lpSum([Vars[f'a_{g}_{i}'] for i in range(N) if i != g])
        mylp += Vars[f'B_{g}'] == pulp.lpSum([Vars[f'b_{g}_{j}'] for j in range(N) if j != g])
        mylp += Vars[f'C_{g}'] == pulp.lpSum([Vars[f'a_{k}_{g}'] for k in range(N) if k != g])
        mylp += Vars[f'D_{g}'] == pulp.lpSum([Vars[f'b_{k}_{g}'] for k in range(N) if k != g])
    
    # 3. Communication Volume
    for g in range(N):
        mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 <= Vars[f'Comm_Volume']
        mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 <= Vars[f'Comm_Volume']
    
    # 4. Load Balance
    COMP_UB = int(math.ceil((1 + N - 1) * (N - 1) / 2 / N))
    for g in range(N):
        mylp += pulp.lpSum([Vars[f'x_{i}_{j}_{g}'] for i in range(N) for j in range(i)]) <= COMP_UB

    # Objective
    mylp += Vars[f'Comm_Volume']
    
    # Solve
    t0 = time.time()
    MSG = 1
    # MSG = 0 # disable msg
    mylp.solve(pulp.PULP_CBC_CMD(msg=MSG, timeLimit=TIME_BUDGET))
    # print(f'after solve !!!', flush=True)
    t1 = time.time()
    print(f'LP solve time: {t1 - t0} s', flush=True)
    
    # print_lp_result
    print_lp_result(N, N, Vars, 'varValue')
        
def Quad_LP_GUROBI(N: int):
    # causal = True, fwd
    # Arguments
    problem_type = 'OPT'
    problem_type = 'SAT'
    LOAD_BALANCE = True
    LOAD_BALANCE = False
    if problem_type == 'SAT':
        TARGET = N // 2 + 1 + (LOAD_BALANCE) - 1
    Var_cat_default = GRB.CONTINUOUS
    Var_cat_default = GRB.INTEGER
    
    # LP Problem
    mylp = gp.Model("Workload_Partition_Allocation_GUROBI")
    # Variables & Bound
    constraints = []
    Quad_Bound = 1 / (N * N)
    Quad_Bound = 1 / (2 * N)
    Vars = dict()
    # Var_cat_default = 'Integer'
    for i in range(N):
        for j in range(i):
            for k in range(N):
                Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
                # # Quadratic Constraints
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) >= 0)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * Vars[f"x_{i}_{j}_{k}"] <= 1)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] - cp.square(Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
                # Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
                # mylp.addConstr(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
    for g in range(N):
        for i in range(N):
            Vars[f'a_{g}_{i}'] = mylp.addVar(vtype=Var_cat_default, name=f'a_{g}_{i}', lb=0, ub=1)
            
    for g in range(N):
        for j in range(N):
            Vars[f'b_{g}_{j}'] = mylp.addVar(vtype=Var_cat_default, name=f'b_{g}_{j}', lb=0, ub=1)
    for g in range(N):
        Vars[f'A_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'A_{g}', lb=0, ub=N-1)
        Vars[f'B_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'B_{g}', lb=0, ub=N-1)
        Vars[f'C_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'C_{g}', lb=0, ub=N-1)
        Vars[f'D_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'D_{g}', lb=0, ub=N-1)
        Vars[f'Cin_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'Cin_{g}', lb=0)
        Vars[f'Cout_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'Cout_{g}', lb=0)
    Vars[f'Comm_Volume'] = mylp.addVar(vtype=Var_cat_default, name=f'Comm_Volume', lb=0)
    
    # Constraints
    # -1.
    if problem_type == 'SAT':
        mylp.addConstr(Vars[f'Comm_Volume'] == TARGET)
    # 0. Workload Partition
    for i in range(N):
        for j in range(i):
            # mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) == 1 # w/o Replication
            # mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) >= 1 # w Replication
            mylp.addConstr(gp.quicksum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) == 1) # w/o Replication
    # 1. a, b
    for g in range(N):
        for i in range(N):
            for j in range(i):
                # mylp += Vars[f'a_{g}_{i}'] >= Vars[f'x_{i}_{j}_{g}']
                mylp.addConstr(Vars[f'a_{g}_{i}'] >= Vars[f'x_{i}_{j}_{g}'])
    for g in range(N):
        for j in range(N):
            for i in range(j + 1, N):
                # mylp += Vars[f'b_{g}_{j}'] >= Vars[f'x_{i}_{j}_{g}']
                mylp.addConstr(Vars[f'b_{g}_{j}'] >= Vars[f'x_{i}_{j}_{g}'])
    
    # 2. A, B, C, D
    for g in range(N):
        # mylp += Vars[f'A_{g}'] == pulp.lpSum([Vars[f'a_{g}_{i}'] for i in range(N) if i != g])
        # mylp += Vars[f'B_{g}'] == pulp.lpSum([Vars[f'b_{g}_{j}'] for j in range(N) if j != g])
        # mylp += Vars[f'C_{g}'] == pulp.lpSum([Vars[f'a_{k}_{g}'] for k in range(N) if k != g])
        # mylp += Vars[f'D_{g}'] == pulp.lpSum([Vars[f'b_{k}_{g}'] for k in range(N) if k != g])
        mylp.addConstr(Vars[f'A_{g}'] == gp.quicksum([Vars[f'a_{g}_{i}'] for i in range(N) if i != g]))
        mylp.addConstr(Vars[f'B_{g}'] == gp.quicksum([Vars[f'b_{g}_{j}'] for j in range(N) if j != g]))
        mylp.addConstr(Vars[f'C_{g}'] == gp.quicksum([Vars[f'a_{k}_{g}'] for k in range(N) if k != g]))
        mylp.addConstr(Vars[f'D_{g}'] == gp.quicksum([Vars[f'b_{k}_{g}'] for k in range(N) if k != g]))
    
    # 3. Communication Volume (In/Out)
    for g in range(N):
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 <= Vars[f'Comm_Volume']
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 <= Vars[f'Comm_Volume']
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 == Vars[f'Cin_{g}'])
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 == Vars[f'Cout_{g}'])
        mylp.addConstr(Vars[f'Cin_{g}'] <= Vars[f'Comm_Volume'])
        mylp.addConstr(Vars[f'Cout_{g}'] <= Vars[f'Comm_Volume'])
    
    # # 4. Load Balance
    if LOAD_BALANCE:
        COMP_UB = int(math.ceil((1 + N - 1) * (N - 1) / 2 / N))
        for g in range(N):
            # mylp += pulp.lpSum([Vars[f'x_{i}_{j}_{g}'] for i in range(N) for j in range(i)]) <= COMP_UB
            mylp.addConstr(gp.quicksum([Vars[f'x_{i}_{j}_{g}'] for i in range(N) for j in range(i)]) <= COMP_UB)

    # Objective
    mylp.setObjective(Vars[f'Comm_Volume'], GRB.MINIMIZE)
    
    # Solve
    t0 = time.time()
    mylp.optimize()
    t1 = time.time()
    print(f'LP solve time: {t1 - t0} s', flush=True)
    
    # # print_lp_result
    if mylp.status == gp.GRB.OPTIMAL:
        print(f"Optimal value: {mylp.objVal}")
    print_lp_result(N, N, Vars, 'x')


    # causal = True, fwd

def Quad_LP_GUROBI_from_block_config(block_config: Block_Attention_Config):
    # fwd
    # Arguments: Begin -----------------------------------------------
    problem_type = 'OPT'
    # problem_type = 'SAT'
    LOAD_BALANCE = True
    # LOAD_BALANCE = False
    if problem_type == 'SAT':
        TARGET = - 1
        # TARGET = N // 2 + 1 + (LOAD_BALANCE) - 1
    Var_cat_default = GRB.CONTINUOUS
    Var_cat_default = GRB.INTEGER
    # Arguments: End -------------------------------------------------
    
    # LP Problem
    mylp = gp.Model("Workload_Partition_Allocation_GUROBI")
    # Variables & Bound
    constraints = []
    # Quad_Bound = 1 / (N * N)
    # Quad_Bound = 1 / (2 * N)
    Vars = dict()
    # Var_cat_default = 'Integer'
    
    ParD = block_config.ParD
    CP = block_config.CP
    cmap = block_config.cmap
    
    block_ids = []  # block_ids to be scheduled
    # Check whether diagonal line is full
    diagonal_full = True    # [NOTE]: haven't consider imbalanced workload on diagonal line in load balance !!!
    for i in range(ParD):
        if block_config.block_table[i, i] == Block_Type.EMPTY:
            diagonal_full = False
            break
    for i in range(ParD):
        for j in range(ParD):
            if i == j and diagonal_full:  # schedule block on diagonal line to cmap[i] by default
                continue
            if block_config.block_table[i, j] != Block_Type.EMPTY:
                block_ids.append((i, j))
    
    for i, j in block_ids:
        for k in range(CP):
            Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
            # # Quadratic Constraints
            # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
            # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) >= 0)
            # constraints.append(Vars[f"x_{i}_{j}_{k}"] * Vars[f"x_{i}_{j}_{k}"] <= 1)
            # constraints.append(Vars[f"x_{i}_{j}_{k}"] - cp.square(Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
            # Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
            # mylp.addConstr(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
    for g in range(CP):
        for i in range(ParD):
            Vars[f'a_{g}_{i}'] = mylp.addVar(vtype=Var_cat_default, name=f'a_{g}_{i}', lb=0, ub=1)
            
    for g in range(CP):
        for j in range(ParD):
            Vars[f'b_{g}_{j}'] = mylp.addVar(vtype=Var_cat_default, name=f'b_{g}_{j}', lb=0, ub=1)
    for g in range(CP):
        Vars[f'A_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'A_{g}', lb=0)
        Vars[f'B_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'B_{g}', lb=0)
        Vars[f'C_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'C_{g}', lb=0)
        Vars[f'D_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'D_{g}', lb=0)
        Vars[f'Cin_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'Cin_{g}', lb=0)
        Vars[f'Cout_{g}'] = mylp.addVar(vtype=Var_cat_default, name=f'Cout_{g}', lb=0)
    Vars[f'Comm_Volume'] = mylp.addVar(vtype=Var_cat_default, name=f'Comm_Volume', lb=0)
    
    # Constraints
    # -1.
    if problem_type == 'SAT':
        mylp.addConstr(Vars[f'Comm_Volume'] == TARGET)
    # 0. Workload Partition
    for i, j in block_ids:
        # mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) == 1 # w/o Replication
        # mylp += pulp.lpSum([Vars[f"x_{i}_{j}_{k}"] for k in range(N)]) >= 1 # w Replication
        mylp.addConstr(gp.quicksum([Vars[f"x_{i}_{j}_{k}"] for k in range(CP)]) == 1) # w/o Replication
    # 1. a, b
    for g in range(CP):
        for i, j in block_ids:
            # mylp += Vars[f'a_{g}_{i}'] >= Vars[f'x_{i}_{j}_{g}']
            mylp.addConstr(Vars[f'a_{g}_{i}'] >= Vars[f'x_{i}_{j}_{g}'])
    for g in range(CP):
        for i, j in block_ids:
            # mylp += Vars[f'b_{g}_{j}'] >= Vars[f'x_{i}_{j}_{g}']
            mylp.addConstr(Vars[f'b_{g}_{j}'] >= Vars[f'x_{i}_{j}_{g}'])
    
    # 2. A, B, C, D
    for g in range(CP):
        # mylp += Vars[f'A_{g}'] == pulp.lpSum([Vars[f'a_{g}_{i}'] for i in range(N) if i != g])
        # mylp += Vars[f'B_{g}'] == pulp.lpSum([Vars[f'b_{g}_{j}'] for j in range(N) if j != g])
        # mylp += Vars[f'C_{g}'] == pulp.lpSum([Vars[f'a_{k}_{g}'] for k in range(N) if k != g])
        # mylp += Vars[f'D_{g}'] == pulp.lpSum([Vars[f'b_{k}_{g}'] for k in range(N) if k != g])
        mylp.addConstr(Vars[f'A_{g}'] == gp.quicksum([Vars[f'a_{g}_{i}'] for i in range(ParD) if cmap[i] != g]))
        mylp.addConstr(Vars[f'B_{g}'] == gp.quicksum([Vars[f'b_{g}_{j}'] for j in range(ParD) if cmap[j] != g]))
        mylp.addConstr(Vars[f'C_{g}'] == gp.quicksum([Vars[f'a_{k}_{i}'] for i in range(ParD) if cmap[i] == g for k in range(CP) if k != g]))
        mylp.addConstr(Vars[f'D_{g}'] == gp.quicksum([Vars[f'b_{k}_{j}'] for j in range(ParD) if cmap[j] == g for k in range(CP) if k != g]))
    
    # 3. Communication Volume (In/Out)
    for g in range(CP):
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 <= Vars[f'Comm_Volume']
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 <= Vars[f'Comm_Volume']
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 == Vars[f'Cin_{g}'])
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 == Vars[f'Cout_{g}'])
        mylp.addConstr(Vars[f'Cin_{g}'] <= Vars[f'Comm_Volume'])
        mylp.addConstr(Vars[f'Cout_{g}'] <= Vars[f'Comm_Volume'])
    
    # # 4. Load Balance
    if LOAD_BALANCE:
        COMP_TOTAL = sum([Block_Comp_Volume[block_config.block_table[i, j]] for i, j in block_ids])
        COMP_UB = int(math.ceil(COMP_TOTAL / CP))
        for g in range(CP):
            # mylp += pulp.lpSum([Vars[f'x_{i}_{j}_{g}'] for i in range(N) for j in range(i)]) <= COMP_UB
            mylp.addConstr(gp.quicksum([Vars[f'x_{i}_{j}_{g}'] for i, j in block_ids]) <= COMP_UB)

    # Objective
    mylp.setObjective(Vars[f'Comm_Volume'], GRB.MINIMIZE)
    
    # Solve
    t0 = time.time()
    mylp.optimize()
    t1 = time.time()
    print(f'LP solve time: {t1 - t0} s', flush=True)
    
    # # print_lp_result
    if mylp.status == gp.GRB.OPTIMAL:
        print(f"Optimal value: {mylp.objVal}")
    print_lp_result(CP, ParD, Vars, 'x', cmap=cmap, diagonal_full=diagonal_full)

    # causal = True, fwd

def solve_global_causal():
    # # 
    # N = 16
    # # ILP(N)
    # # Quad_LP_CVXPY(N)
    # # Quad_LP_SCIPY(N)
    # Quad_LP_GUROBI(N)
    CP, Par_D = 2, 4
    # CP, Par_D = 4, 8
    # Par_D = 8
    cmap = np.array([i // (Par_D // CP) for i in range(Par_D)]) # (0, 0, 1, 1, ..., CP-1, CP-1)
    block_config = Block_Attention_Config.from_causal(CP, Par_D, cmap)
    Quad_LP_GUROBI_from_block_config(block_config)
    
    
def solve_custom_sparse():
    # # star
    # CP, Par_D = 4, 8
    # pattern_type, pattern_sparsity, local_blocks = "star", 0.25, 1
    # pattern_type, pattern_sparsity, local_blocks = "stream", 0.25, 2
    # block_config = create_block_sparse_pattern(CP, Par_D, pattern_type, pattern_sparsity, local_blocks)
    # Quad_LP_GUROBI_from_block_config(block_config)
    # local_global (stride_remap_pattern)
    replicate = 1   # default
    
    CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 4, 8, "local_global", 1 / 4, 2, 0, 1
    CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 4, 16, "local_global", 1 / 16, 1, 1, 1
    CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 4, 16, "local_global", 1 / 16, 2, 1, 1
    
    # 1. For stride(1/16, 4, 3):
    # 1.1 for stride(1/16, 4, 3) 8x8 (?/1)
    CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 3, 0, 2
    # # 1.2 for stride(1/16, 4, 3) 8x4 (?/1)
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 3, 0
    # # 1.3 for stride(1/16, 4, 3) 8x8 (?/3)
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, 2, 0, 1  # full
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, (2, 1), 0, 1  # lower triangle
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, (1, 2), 0, 1  # higher triangle
    
    # 2. For local+global(1/16, 1, 1):
    # 2.1 for local+global(1/16, 1, 1) 8x2 (?/4)
    CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 8, 1, 1, 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 8, 0, (1, 0), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 8, 0, (0, 1), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 8, 1, 0, 1
    # # # 2.2 for local+global(1/16, 1, 1) 8x4 (?/4)
    # # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 1, 1, 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 0, (1, 0), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 0, (0, 1), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 4, 1, 0, 1
    # # # 2.3 for local+global(1/16, 1, 1) 8x8 (?/4)
    # # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, 2, 0, 1  # full
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, 0, (1, 0), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, 0, (0, 1), 1
    # CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate = 8, 8, "local_global", 1 / 2, 1, 0, 1
    
    block_config = create_block_sparse_pattern(CP, Par_D, pattern_type, pattern_sparsity, local_blocks, global_blocks, replicate)
    Quad_LP_GUROBI_from_block_config(block_config)

  
def main():
    # solve_global_causal()
    solve_custom_sparse()

    
if __name__ == '__main__':
    main()