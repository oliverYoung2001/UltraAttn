import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.dependent_graph import Dependent_Graph, Cuda_Kernel, Comp_Kernel
import pulp
import regex as re
import random
import json
import time
from heapq import heappush, heappop, heappushpop
import numpy as np
from search_algo.utils import print_rank_0
import gurobipy as gp
from gurobipy import GRB
from typing import Optional

class Fused_Execution_Plan():
    def __init__(self, Y: int, X: int, time: float, causal: bool, fob: bool, ):
        self.Y = Y
        self.X = X
        self.time = time
        assert causal == False, "[Error]: causal attention is supported in Fused_Execution_Plan"
        self.causal = causal
        self.fob = fob
        self.end_time = time    # [TODO]
        
    def __str__(self):
        ret = f'Y={self.Y}, X={self.X}, fused={True}, causal={self.causal}, time={self.time}, fob={self.fob}'
        ret = ret.replace(' ', '')
        return ret
        
class Execution_Plan(): # input: kernel streams of gpus
    def __init__(self, d_graph: Dependent_Graph, fob: bool, plan_type: Optional[str], is_hack: bool = False):
        # self.stream_kernel_lists  # (hierarchy_sp, 3) -> list, 3 stands for comp, send, recv
        # self.gpu_kernel_lists
        # self.valid_kernels    # 
        # self.stream_num   # 3
        
        if not is_hack:
            self.fob = fob  # fwd or bwd
            self.plan_type = plan_type
            assert plan_type in ['ILP', 'Flexflow', None]
            self.d_graph = d_graph
            self.da_config = d_graph.da_config
            self.m_config = d_graph.m_config
            self.split_degrees = d_graph.split_degrees
            self.tot_sp = d_graph.tot_sp
            self.hierarchy = d_graph.hierarchy
            self.hierarchy_sp = self.da_config.SP[self.hierarchy]
            if plan_type == 'ILP':
                self.TIME_BUDGET = 5 * 60   # 5mins
                self.TIME_BUDGET = 1 * 60   # 1mins
                # self.TIME_BUDGET = 10   # 10s
                # self.threshold = 1.3
                self.generate_execution_plan()
            elif plan_type == 'Flexflow':  # Flexflow
                self.generate_execution_plan_through_start_time()
        else:
            self.fob = fob
            self.plan_type = plan_type
            assert plan_type in ['ILP', 'Flexflow']
            self.d_graph = None
            self.da_config = None
            self.m_config = None
            self.split_degrees = (1, 1, 1, 1)
            self.tot_sp = None
            self.hierarchy = 0  # 0 stands for inter, 1 stands for intra
            self.hierarchy_sp = 1
            # [TODO]: Hack gpu_kernel_lists, stream_num, valid_kernels
            self.stream_num = 3
            # dict keys: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids), (c_ids), gpuid)
            kernel_key = (0, 0, 0, 0, 0)
            only_kernel = Comp_Kernel(
                kernel_key, None, None, 
                self.hierarchy, time=np.array([1e-9, 1e-9], dtype=np.float32))
            self.gpu_kernel_lists = [[only_kernel]]
            self.valid_kernels = [only_kernel]
    
    @classmethod
    def create_one_node_exe_plan(cls, fob: bool):
        return cls(None, fob, plan_type='ILP', is_hack=True)
        
    def get_plan_name(self):
        if self.plan_type == 'manual':
            return f'SP={self.hierarchy_sp}_fob={self.fob}_Y={self.Y}_X={self.X}_dim={self.first_dim}'
        else:
            da_config = self.da_config
            return da_config.get_plan_name(self.fob)
        return None
    
    def solve_ILP_with_gurobipy(self, TOT_TIME_UP):
        fob = self.fob
        d_graph = self.d_graph
        # LP Problem
        mylp = gp.Model("Kernel_Schedule_ILP_GUROBI")
        mylp.setParam('OutputFlag', 0)  # [NOTE]: disable output of gurobi
        CPUS_NUM = int(os.getenv('GUROBI_NUM_THREADS', default=64))
        mylp.setParam('Threads', CPUS_NUM)
        mylp.setParam('TimeLimit', self.TIME_BUDGET)
        mylp.setParam('Seed', 42)
        # End
        
        # Variables
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                v.start_time = mylp.addVar(vtype=GRB.CONTINUOUS, name=f"start_time[{v.id}]", lb=0)
                # v.start_time = pulp.LpVariable(f'start_time[{v.id}]', cat=pulp.const.LpContinuous, lowBound=0)
        # end_time = pulp.LpVariable(f'end_time', cat=pulp.const.LpContinuous, lowBound=0)
        end_time = mylp.addVar(vtype=GRB.CONTINUOUS, name=f'end_time', lb=0)
        
        # Contrains
        #   1. stream exclusive
        overlap_set = set()
        for kernel_list_key, kernel_list in self.stream_kernel_lists.items():
            # print_rank_0(f'kernel_list_key: {kernel_list_key}') # (0, 1, 2) -> comp, send, recv
            for i in range(len(kernel_list)):
                for j in range(i + 1, len(kernel_list)):
                    id0 = kernel_list[i].id
                    id1 = kernel_list[j].id
                    overlap_key = (min(id0, id1), max(id0, id1))
                    if overlap_key in overlap_set:
                        continue
                    # print_rank_0(f'overlap_key: {overlap_key}; kernel keys: {kernel_list[i].key}, {kernel_list[j].key}')
                    overlap_set.add(overlap_key)
                    # overlap_ij = pulp.LpVariable(f"overlap_{overlap_key[0]}_{overlap_key[1]}", cat='Binary')
                    overlap_ij = mylp.addVar(vtype=GRB.BINARY, name=f"overlap_{overlap_key[0]}_{overlap_key[1]}")
                    # mylp += kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time + TOT_TIME_UP * overlap_ij
                    # mylp += kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time + TOT_TIME_UP * (1 - overlap_ij)
                    mylp.addConstr(kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time + TOT_TIME_UP * overlap_ij)
                    mylp.addConstr(kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time + TOT_TIME_UP * (1 - overlap_ij))
                    
        #   2. kernel dependences
        for u in self.valid_kernels:
            for v in u.successors:
                # mylp += v.start_time >= u.start_time + u.time[fob]
                mylp.addConstr(v.start_time >= u.start_time + u.time[fob])
        
        #   3. end_time
        for u in self.valid_kernels:
            # mylp += end_time >= u.start_time + u.time[fob]
            mylp.addConstr(end_time >= u.start_time + u.time[fob])
        
        #   Objective
        # mylp += end_time 
        mylp.setObjective(end_time, GRB.MINIMIZE) 
                    
        # Solve
        t0 = time.time()
        mylp.optimize()
        t1 = time.time()
        print(f'LP solve time for {mylp.ModelName}: {t1 - t0} s', flush=True)
        
        # parse end_time and start_time of each kernel
        for v in self.valid_kernels:
            v._start_time = v.start_time.x
        self.end_time = self.mylp_obj = mylp.objVal

    def solve_ILP_with_pulp(self, TOT_TIME_UP):
        fob = self.fob
        d_graph = self.d_graph
        # LP Problem
        # mylp = gp.Model("Kernel_Schedule_ILP_GUROBI")
        # mylp.setParam('OutputFlag', 0)  # [NOTE]: disable output of gurobi
        # CPUS_NUM = int(os.getenv('GUROBI_NUM_THREADS', default=64))
        # mylp.setParam('Threads', CPUS_NUM)
        # mylp.setParam('TimeLimit', self.TIME_BUDGET)
        # mylp.setParam('Seed', 42)
        mylp = pulp.LpProblem(f"Kernel_Schedule_ILP_PULP", pulp.LpMinimize)
        # End
        
        # Variables
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                v.start_time = pulp.LpVariable(f'start_time[{v.id}]', cat=pulp.const.LpContinuous, lowBound=0)
        end_time = pulp.LpVariable(f'end_time', cat=pulp.const.LpContinuous, lowBound=0)
        
        # Contrains
        #   1. stream exclusive
        overlap_set = set()
        for kernel_list_key, kernel_list in self.stream_kernel_lists.items():
            # print_rank_0(f'kernel_list_key: {kernel_list_key}') # (0, 1, 2) -> comp, send, recv
            for i in range(len(kernel_list)):
                for j in range(i + 1, len(kernel_list)):
                    id0 = kernel_list[i].id
                    id1 = kernel_list[j].id
                    overlap_key = (min(id0, id1), max(id0, id1))
                    if overlap_key in overlap_set:
                        continue
                    # print_rank_0(f'overlap_key: {overlap_key}; kernel keys: {kernel_list[i].key}, {kernel_list[j].key}')
                    overlap_set.add(overlap_key)
                    overlap_ij = pulp.LpVariable(f"overlap_{overlap_key[0]}_{overlap_key[1]}", cat='Binary')
                    mylp += kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time + TOT_TIME_UP * overlap_ij
                    mylp += kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time + TOT_TIME_UP * (1 - overlap_ij)
                    
        #   2. kernel dependences
        for u in self.valid_kernels:
            for v in u.successors:
                mylp += v.start_time >= u.start_time + u.time[fob]
        
        #   3. end_time
        for u in self.valid_kernels:
            mylp += end_time >= u.start_time + u.time[fob]
        
        #   Objective
        mylp += end_time 
                    
        # Solve
        t0 = time.time()
        MSG = 1
        # MSG = 0 # disable msg
        CPUS_NUM = int(os.getenv('GUROBI_NUM_THREADS', default=64))
        solver = pulp.GUROBI(msg=MSG, timeLimit=self.TIME_BUDGET, Seed=42)    # Cannot pass sanity check !!!
        # solver = pulp.CPLEX_CMD(msg=MSG, timeLimit=self.TIME_BUDGET, threads=CPUS_NUM, options=[("randomseed", 42)])    # Need CPLEX
        # solver = pulp.PULP_CBC_CMD(msg=MSG, timeLimit=self.TIME_BUDGET, threads=CPUS_NUM, options=["randomSeed", "42"])   # Too slow
        # solver = pulp.SCIP_CMD(msg=MSG, timeLimit=self.TIME_BUDGET, threads=CPUS_NUM, options=[("randomization/randomseed", 42)]) # Need SCIP
        # solver = pulp.CHOCO_CMD(msg=MSG, timeLimit=self.TIME_BUDGET)    # Need java
        # solver = pulp.GLPK_CMD(msg=MSG, timeLimit=self.TIME_BUDGET) # Need GLPK
        mylp.solve(solver)    # Use GUROBI !!!
        t1 = time.time()
        print(f'LP solve time for {mylp.name}: {t1 - t0} s', flush=True)
        
        # parse end_time and start_time of each kernel
        for v in self.valid_kernels:
            v._start_time = v.start_time.value()
        self.end_time = self.mylp_obj = pulp.value(mylp.objective)

    def generate_execution_plan(self):
        fob = self.fob
        d_graph = self.d_graph
        # print_rank_0(f'TOT_TIME_UP: {TOT_TIME_UP}')
        # Using LP to optimize the execution order
        self.valid_kernels = []
        v_id = 0
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                self.valid_kernels.append(v)
                v.id = v_id
                v_id += 1
                # v.start_time = pulp.LpVariable(f'start_time[{v.id}]', cat=pulp.const.LpContinuous, lowBound=0)
        # end_time = pulp.LpVariable(f'end_time', cat=pulp.const.LpContinuous, lowBound=0)
        
        self.stream_num = 3
        stream_kernel_lists = {}  # (hierarchy_sp, 3) -> list, 3 stands for comp, send, recv
        self.stream_kernel_lists = stream_kernel_lists
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                stream_kernel_lists[(g, s)] = []
        for v in self.valid_kernels:
            if v.type == 'comp':
                stream_kernel_lists[(v.key[-1], 0)].append(v)
            elif v.type == 'comm':
                stream_kernel_lists[(v.key[3], 1)].append(v)
                stream_kernel_lists[(v.key[4], 2)].append(v)
        
        # Build&Solve ILP
        # TOT_TIME_UP ⬆️ -> solver performance ⬆️; corretness ⬇️
        # TOT_TIME_UP = d_graph.schedule.get_e2e_time()[fob] * 3  # Significant for GUROBI
        # TOT_TIME_UP = d_graph.schedule.get_e2e_time()[fob] * 8  # Significant for GUROBI
        # TOT_TIME_UP = d_graph.schedule.get_absolute_cc_time()
        # TOT_TIME_UP = 1e9
        TOT_TIME_UP = sum([v.time[fob] for v in self.valid_kernels])    # Sum of all kernels
        print_rank_0(f'TOT_TIME_UP: {TOT_TIME_UP}')
        self.solve_ILP_with_gurobipy(TOT_TIME_UP)   # with gurobipy !!!
        # self.solve_ILP_with_pulp(TOT_TIME_UP)    # with pulp(gurobi)
        
        # Sort all kernels in each stream according to start_time (kernel orders on all stream are determined !!!)
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                self.stream_kernel_lists[(g, s)].sort(key=lambda x: x._start_time)
        # self.print_lp_result()
        # Fine-tune start_time according to kernel orders
        self.fine_tune_kernel_time()
        
        # Sanity check for stream exclusive on stream_kernel_lists
        if not self.sanity_check_stream_exclusive():
            print_rank_0(f'[ERROR]: sanity check for stream exclusive not passed !!!')
        else:
            print_rank_0(f'[INFO]: sanity check for stream exclusive passed.')
        
        self.gpu_kernel_lists = []
        print_rank_0(f'gpu_kernel_lists:')
        for g in range(self.hierarchy_sp):
            kernel_list = []
            for s in range(self.stream_num):
                kernel_list += self.stream_kernel_lists[(g, s)]
            # kernel_list.sort(key=lambda x: (x._start_time, x.id))
            kernel_list.sort(key=lambda x: (x._start_time, isinstance(x, Comp_Kernel)))
            self.gpu_kernel_lists.append(kernel_list)
            # Print gpu_kernel_lists
            print_rank_0(f'G{g} {" ".join([str(kernel.key) for kernel in kernel_list])}')
            
    def fine_tune_kernel_time(self):
        # 1. Correct overlapped kernels on the same stream caused by GUROBI
        # 2. Minimize start times for all kernel under constraints
        fob = self.fob
        # search according to topology order
        # bfs to calc start_time of kernels
        def pack_func(v: Cuda_Kernel) -> tuple:
            return (v._start_time, v.id, v)
        def unpack_func(q_item: tuple) -> Cuda_Kernel:
            return q_item[2]
        pq = []     # Priprity Queue
        
        # build dependencies within streams
        for kernel_list in self.stream_kernel_lists.values():   
            for i in range(len(kernel_list) - 1):
                # assert kernel_list[i + 1] not in kernel_list[i].successors, \
                #     f'[ERROR]: {kernel_list[i + 1].key} should not be in successors of {kernel_list[i].key}'
                # assert kernel_list[i] not in kernel_list[i + 1].precursors, \
                #     f'[ERROR]: {kernel_list[i].key} should not be in precursors of {kernel_list[i + 1].key}'
                # kernel_list[i].successors.add(kernel_list[i + 1])
                # kernel_list[i + 1].precursors.add(kernel_list[i])
                kernel_list[i].add_edge(kernel_list[i + 1], fob)
                
        # calc initialize kernels
        for v in self.valid_kernels:
            v.left_precursors = len(v.precursors)
            v._start_time = 0
            v.selected = False
            if v.left_precursors == 0:
                heappush(pq, pack_func(v))
        while len(pq) > 0:
            v = unpack_func(heappop(pq))    # select a kernel
            v.selected = True
            # # update start_times of kernels in the same streams with v
            # for stream_key in v.stream_keys:
            #     for u in self.stream_kernel_lists[stream_key]:
            #         if not u.selected:
            #             u._start_time = max(u._start_time, v._start_time + v.time[fob])
            # update start_times of successive kernels of v
            for u in v.successors:
                u.left_precursors -= 1
                u._start_time = max(u._start_time, v._start_time + v.time[fob])
                if u.left_precursors == 0:
                    heappush(pq, pack_func(u))
                    
        # remove dependencies within streams; in reverse order !!!!
        for kernel_list in reversed(self.stream_kernel_lists.values()):   
            for i in range(len(kernel_list) - 2, - 1, - 1):
                kernel_list[i].remove_edge(kernel_list[i + 1], fob)
                
        # Recompute correct end_time
        self.end_time = 0
        for v in self.valid_kernels:
            self.end_time = max(self.end_time, v._start_time + v.time[fob])
        
    def sanity_check_stream_exclusive(self):    # Assume that kernels on each list are sorted by their start time
        fob = self.fob
        ret = True
        THRESHOLD = 1e-3
        for kernel_list_key, kernel_list in self.stream_kernel_lists.items():
            for i in range(len(kernel_list) - 1):
                if (kernel_list[i]._start_time + kernel_list[i].time[fob] - kernel_list[i + 1]._start_time) / kernel_list[i].time[fob] \
                    > THRESHOLD:
                    print_rank_0(f'Sanity check Error: {kernel_list_key}, {i}th; '
                                 f'{kernel_list[i]._start_time:.3e} + {kernel_list[i].time[fob]:.3e} = '
                                 f'{(kernel_list[i]._start_time + kernel_list[i].time[fob]):.3e} '
                                 f'> {kernel_list[i + 1]._start_time:.3e}')
                    ret = False
        return ret
        
    def get_end_time(self):
        return self.end_time if hasattr(self, 'end_time') else self.mylp_obj
    
    def print_lp_result(self):
        fob = self.fob
        d_graph = self.d_graph
        hierarchy = d_graph.hierarchy
        hierarchy_sp = self.hierarchy_sp
        OJB = 'node' if hierarchy == 0 else 'gpu'
        
        print(f'fob: {fob}; schedule:\n{d_graph.schedule.schedule_table}', flush=True)
        # print(f'get_e2e_time(): {d_graph.schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{d_graph.schedule.get_absolute_cc_time()[fob]}', flush=True)
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                print(f'{v.key}: {v._start_time:.3e}, {v.time[fob]:.3e}, {(v._start_time + v.time[fob]):.3e}')
        
        print(f'Streams:')
        for g in range(hierarchy_sp):
            for s in range(3):
                print(f"{OJB}{g}, {['comp', 'send', 'recv'][s]}: {len(self.stream_kernel_lists[(g, s)])}")
                for v in self.stream_kernel_lists[(g, s)]:
                    print(f'{v.key}: {v._start_time:.3e}, {v.time[fob]:.3e}, {(v._start_time + v.time[fob]):.3e}')
        # if self.plan_type == 'ILP':
        #     print(f'objective={self.mylp_obj:.3e}', flush=True)
        # # elif self.plan_type == 'manual':
        # else:
        print(f'end_time={self.end_time:.3e}', flush=True)
    
    def determine_kernel_order_for_full(self):   # [NOTE]: for full attn intra-node, non-fused, manually cc schedule
        hierarchy_sp = self.hierarchy_sp
        X = self.X
        Y = self.Y
        first_dim = self.first_dim
        fob = self.fob
        split_degrees = self.split_degrees
        d_graph = self.d_graph
        # self.stream_kernel_lists  # (hierarchy_sp, 3) -> list, 3 stands for comp, send, recv
        self.stream_num = 3
        stream_kernel_lists = {}
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                stream_kernel_lists[(g, s)] = []
                # [TODO]:
                
        def calc_rank_from_offset(rank: tuple, offset: tuple) -> tuple:
            # rank: (Y, X)
            # rank: ([rank[0] // X * X, (rank[0] // X + 1 * X), rank[0] % X + [0, Y) * X)
            lb0 = rank[0] // X * X
            nr0 = rank[0] + offset[1]
            nr1 = rank[1] + offset[0] * X
            return (lb0 + ((nr0 - lb0) % X + X) % X, (nr1 % hierarchy_sp + hierarchy_sp) % hierarchy_sp)
        # def get_tuple_rank(g: int) -> tuple:
        #     return (g // X, g % X)
        # def get_real_rank(rank: tuple) -> int:
        #     return rank[0] * X + rank[1]
            
        assert split_degrees[0] == split_degrees[1] == hierarchy_sp and split_degrees[0] % X == 0
        assert self.da_config.causal == False, "[Error]: causal attention is supported in manually designed plans"
        
        for b_id in range(split_degrees[2]):
            for h_id in range(split_degrees[3]):
                # Comp orders
                for g in range(self.hierarchy_sp):
                    kernel_list = stream_kernel_lists[(g, 0)]
                    kernel_list_send = stream_kernel_lists[(g, 1)]
                    kernel_list_recv = stream_kernel_lists[(g, 2)]
                    init_rc_ids = (g, g)
                    if first_dim == 0:
                        for off0 in range(Y):
                            for off1 in range(X):
                                cur_rc_ids = calc_rank_from_offset(init_rc_ids, (off0, off1))
                                key = (b_id, h_id, *cur_rc_ids, g)
                                kernel_list.append(d_graph.kernel_dict[key])
                        # Comm orders: strategy: all input are prior to all output
                        # ir
                        for off1 in range(1, X):
                            recv_rc_ids = calc_rank_from_offset(init_rc_ids, (0, off1))
                            send_rc_ids = calc_rank_from_offset(init_rc_ids, (0, - off1))
                            key_recv = (b_id, h_id, recv_rc_ids[0], recv_rc_ids[0], g, 'i', 'r')
                            key_send = (b_id, h_id, g, g, send_rc_ids[0], 'i', 'r')
                            kernel_list_recv.append(d_graph.kernel_dict[key_recv])
                            kernel_list_send.append(d_graph.kernel_dict[key_send])
                        # ic
                        for off0 in range(1, Y):
                            recv_rc_ids = calc_rank_from_offset(init_rc_ids, (off0, 0))
                            send_rc_ids = calc_rank_from_offset(init_rc_ids, (- off0, 0))
                            key_recv = (b_id, h_id, recv_rc_ids[1], recv_rc_ids[1], g, 'i', 'c')
                            key_send = (b_id, h_id, g, g, send_rc_ids[1], 'i', 'c')
                            kernel_list_recv.append(d_graph.kernel_dict[key_recv])
                            kernel_list_send.append(d_graph.kernel_dict[key_send])
                        # oc
                        for off0 in range(1, Y - 1):
                            send_rc_ids = calc_rank_from_offset(init_rc_ids, (off0, 0))
                            recv_rc_ids = calc_rank_from_offset(init_rc_ids, (- off0, 0))
                            key_send = (b_id, h_id, send_rc_ids[1], g, send_rc_ids[1], 'o', 'c')
                            key_recv = (b_id, h_id, g, recv_rc_ids[1], g, 'o', 'c')
                            kernel_list_send.append(d_graph.kernel_dict[key_send])
                            kernel_list_recv.append(d_graph.kernel_dict[key_recv])
                        # or
                        for off1 in range(1, X):
                            send_rc_ids = calc_rank_from_offset(init_rc_ids, (0, off1))
                            recv_rc_ids = calc_rank_from_offset(init_rc_ids, (0, - off1))
                            key_send = (b_id, h_id, send_rc_ids[0], g, send_rc_ids[0], 'o', 'r')
                            key_recv = (b_id, h_id, g, recv_rc_ids[0], g, 'o', 'r')
                            kernel_list_send.append(d_graph.kernel_dict[key_send])
                            kernel_list_recv.append(d_graph.kernel_dict[key_recv])
                        # last oc
                        off0 = Y - 1
                        if off0 >= 1:
                            send_rc_ids = calc_rank_from_offset(init_rc_ids, (off0, 0))
                            recv_rc_ids = calc_rank_from_offset(init_rc_ids, (- off0, 0))
                            key_send = (b_id, h_id, send_rc_ids[1], g, send_rc_ids[1], 'o', 'c')
                            key_recv = (b_id, h_id, g, recv_rc_ids[1], g, 'o', 'c')
                            kernel_list_send.append(d_graph.kernel_dict[key_send])
                            kernel_list_recv.append(d_graph.kernel_dict[key_recv])
                        
                    else:
                        assert f"[ERROR]: Not support first_dim={first_dim}"
                        for off1 in range(X):
                            for off0 in range(Y):
                                cur_rc_ids = calc_rank_from_offset(init_rc_ids, (off0, off1))
                                key = (b_id, h_id, *cur_rc_ids, g)
                                kernel_list.append(d_graph.kernel_dict[key])
                        # Comm orders: strategy: all input are prior to all output
                        # ic
                        
                        # ir
                                
                        # or
                        
                        # oc
        # filter out empty kernels
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                kernel_list = stream_kernel_lists[(g, s)]
                for i in range(len(kernel_list) - 1, -1, -1):   # reverse order
                    if kernel_list[i].is_empty(fob):
                        kernel_list.pop(i)
        
        # add extra dependences for kernels in the same stream
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                kernel_list = stream_kernel_lists[(g, s)]
                for i in range(len(kernel_list) - 1):
                    kernel_list[i].add_edge(kernel_list[i + 1], fob)
    
    def determine_kernel_order_by_bfs(self):
        # self.stream_kernel_lists  # (hierarchy_sp, 3) -> list, 3 stands for comp, send, recv
        self.stream_num = 3
        stream_kernel_lists = {}
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                stream_kernel_lists[(g, s)] = []
                # [TODO]:
    
    def generate_execution_plan_through_start_time(self):
        d_graph = self.d_graph
        fob = self.fob
        
        # self.valid_kernels
        if not hasattr(self, 'valid_kernels'):
            self.valid_kernels = []
            v_id = 0
            for v in d_graph.kernel_dict.values():
                if not v.is_empty(fob):
                    self.valid_kernels.append(v)
                    v.id = v_id
                    v_id += 1
        
        # self.stream_num
        if not hasattr(self, 'stream_num'):
            self.stream_num = 3
        # self.stream_kernel_lists  # (hierarchy_sp, 3) -> list, 3 stands for comp, send, recv
        if not hasattr(self, 'stream_kernel_lists'):
            # initialize stream_kernel_lists
            stream_kernel_lists = {}
            for g in range(self.hierarchy_sp):
                for s in range(self.stream_num):
                    stream_kernel_lists[(g, s)] = []
            # insert valid kernels into stream_kernel_lists
            for v in self.valid_kernels:
                if v.type == 'comp':
                    stream_kernel_lists[(v.key[-1], 0)].append(v)
                    v.stream_keys = ((v.key[-1], 0),)
                elif v.type == 'comm':
                    stream_kernel_lists[(v.key[3], 1)].append(v)
                    stream_kernel_lists[(v.key[4], 2)].append(v)
                    v.stream_keys = ((v.key[3], 1), (v.key[4], 2))
            self.stream_kernel_lists = stream_kernel_lists
        
        
        # bfs to calc start_time of kernels
        def pack_func(v: Cuda_Kernel) -> tuple:
            return (v._start_time, v.id, v)
        def unpack_func(q_item: tuple) -> Cuda_Kernel:
            return q_item[2]
        pq = []     # Priprity Queue
        for v in self.valid_kernels:
            v.left_precursors = len(v.precursors)
            v._start_time = 0
            v.selected = False
            if v.left_precursors == 0:
                heappush(pq, pack_func(v))
        while len(pq) > 0:
            v = unpack_func(heappop(pq))    # select a kernel
            v.selected = True
            # update start_times of kernels in the same streams with v
            for stream_key in v.stream_keys:
                for u in self.stream_kernel_lists[stream_key]:
                    if not u.selected:
                        u._start_time = max(u._start_time, v._start_time + v.time[fob])
            for u in v.successors:
                u.left_precursors -= 1
                u._start_time = max(u._start_time, v._start_time + v.time[fob])
                if u.left_precursors == 0:
                    heappush(pq, pack_func(u))
        # calc end_time
        self.end_time = 0
        for v in self.valid_kernels:
            self.end_time = max(self.end_time, v._start_time + v.time[fob])
        
        # sort all kernels in each stream according to start_time
        assert hasattr(self, 'stream_kernel_lists')
        for g in range(self.hierarchy_sp):
            for s in range(self.stream_num):
                self.stream_kernel_lists[(g, s)].sort(key=lambda x: x._start_time)
            
        # self.gpu_kernel_lists
        self.gpu_kernel_lists = []
        for g in range(self.hierarchy_sp):
            kernel_list = []
            for s in range(self.stream_num):
                kernel_list += self.stream_kernel_lists[(g, s)]
            kernel_list.sort(key=lambda x: (x._start_time, x.id))
            self.gpu_kernel_lists.append(kernel_list)
        # calc end_time
        self.end_time = 0
        for v in self.valid_kernels:
            self.end_time = max(self.end_time, v._start_time + v.time[fob])
                
    def generate_manual_plan(self, hierarchy_sp: int, X: int, first_dim: int = 0):
        self.plan_type = 'manual'
        self.hierarchy_sp = hierarchy_sp
        self.X = X
        Y = hierarchy_sp // X
        self.Y = Y
        self.first_dim = first_dim
        print(f'X={X}, Y={Y}, first_dim={first_dim}')
        fob = self.fob
        d_graph = self.d_graph
        # self.valid_kernels
        self.valid_kernels = []
        v_id = 0
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                self.valid_kernels.append(v)
                v.id = v_id
                v_id += 1
        self.determine_kernel_order_for_full()
        self.generate_execution_plan_through_start_time()
    
    def generate_plan_with_one_topological_order(self):
        self.plan_type = 'Flexflow'
        self.generate_execution_plan_through_start_time()
            