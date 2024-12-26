import pulp
import time
import math
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

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

def print_lp_result(N: int, Vars: dict, v_attr: str):
    print(f'LP result:', flush=True)
    for k, v in Vars.items():
        assert hasattr(v, v_attr)
        print(f'{k}: {getattr(v, v_attr)}')
    print(f'Allocation Matrix:')
    for i in range(N):
        for j in range(N):
            if i < j:
                print(f'/', end=' ')
                continue
            if i == j:
                print(f'{i}', end=' ')
                continue
            max_v = 0
            max_kid = -1
            for k in range(N):
                cur_v = getattr(Vars[f'x_{i}_{j}_{k}'], v_attr)
                if cur_v > max_v:
                    max_v = cur_v
                    max_kid = k
            print(f'{max_kid}', end=' ')
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
    print_lp_result(N, Vars, 'varValue')
        
def Quad_LP_GUROBI(N: int):
    # causal = True, fwd
    mylp = gp.Model("Workload_Partition_Allocation_GUROBI")

    # Variables & Bound
    constraints = []
    Quad_Bound = 1 / (N * N)
    Vars = dict()
    Var_cat_default = GRB.CONTINUOUS
    # Var_cat_default = GRB.INTEGER
    # Var_cat_default = 'Integer'
    for i in range(N):
        for j in range(i):
            for k in range(N):
                # Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
                # # Quadratic Constraints
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) >= 0)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] * Vars[f"x_{i}_{j}_{k}"] <= 1)
                # constraints.append(Vars[f"x_{i}_{j}_{k}"] - cp.square(Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
                Vars[f"x_{i}_{j}_{k}"] = mylp.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}_{k}", lb=0, ub=1)
                mylp.addConstr(Vars[f"x_{i}_{j}_{k}"] * (1 - Vars[f"x_{i}_{j}_{k}"]) <= Quad_Bound)
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
    Vars[f'Comm_Volume'] = mylp.addVar(vtype=Var_cat_default, name=f'Comm_Volume', lb=0)
    
    # Constraints
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
    
    # 3. Communication Volume
    for g in range(N):
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 <= Vars[f'Comm_Volume']
        # mylp += Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 <= Vars[f'Comm_Volume']
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'B_{g}'] * 2 <= Vars[f'Comm_Volume'])
        mylp.addConstr(Vars[f'A_{g}'] * 1 + Vars[f'C_{g}'] * 1 + Vars[f'D_{g}'] * 2 <= Vars[f'Comm_Volume'])
    
    # 4. Load Balance
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
    print_lp_result(N, Vars, 'x')


    # causal = True, fwd

def main():
    N = 8
    # ILP(N)
    # Quad_LP_CVXPY(N)
    # Quad_LP_SCIPY(N)
    Quad_LP_GUROBI(N)

    
if __name__ == '__main__':
    main()