import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, Dist_Attn_Config, Machine_Config, Evaluation_Configs
from search_algo.utils import get_factors, print_rank_0
from search_algo.bsa_utils import bsa_is_dense, bsa_is_causal
from search_algo.dependent_graph import Dependent_Graph, Comp_Kernel
from search_algo.execute_plan import Execution_Plan
from search_algo.bsa_config import BSA_Config
import numpy as np
import copy
import itertools
from typing import Optional, List
import math

class Comm_Rebuild_Engine():   # (Broadcast/reduce, row/col)
    def __init__(self):
        pass

class Graph_Substitution():
    def __init__(self):
        pass
    
class Graph_Transformation():
    def __init__(self):
        pass


class Comp_Fusion_Transformation(Graph_Transformation):
    def __init__(self, sub, ids_tuple: tuple):
        super().__init__()
        self.sub = sub
        assert len(ids_tuple) == len(sub.shape)
        for dim in range(len(sub.shape)):
            assert len(ids_tuple[dim]) == sub.shape[dim]
            assert isinstance(ids_tuple[dim], np.ndarray)
        self.ids_tuple = ids_tuple
        self.ids_set = set()
        assert len(ids_tuple) == 2
        for x in ids_tuple[0]:
            for y in ids_tuple[1]:
                self.ids_set.add((x, y))
    
    def apply_on_d_graph(self, d_graph: Dependent_Graph): # ✅
        # [TODO]: apply transformation on d_graph
        da_config: Dist_Attn_Config = d_graph.da_config
        schedule = d_graph.schedule
        hierarchy = d_graph.hierarchy
        bsa_comp_key_suffixes = d_graph.bsa_comp_key_suffixes
        ids_tuple = self.ids_tuple
        ids_set = self.ids_set
        if hierarchy == 0:
            bsa_intra_unit_shape_config = copy.deepcopy(da_config.shape_config)
            bsa_intra_unit_shape_config['S'] = (bsa_intra_unit_shape_config['S'][0] // schedule.split_degrees[0] * len(ids_tuple[0]), \
                                                bsa_intra_unit_shape_config['S'][1] // schedule.split_degrees[1] * len(ids_tuple[1]))
            bsa_CP = (da_config.SP[1], 1)  # (intra, inter)
            sub_bsa_repr = da_config.bsa_config.bsa_repr.create_sub_bsa_repr(
                schedule.split_degrees[0: 2], select_ids = [list(ids_tuple[0]), list(ids_tuple[1])])
            sub_pat_bsa_repr = {'bsa_repr': sub_bsa_repr, 'CP': bsa_CP}
            sub_bsa_config = BSA_Config(None, None, sub_pat_bsa_repr)
            comp_map_key = d_graph.select_bsa_comp_key(bsa_CP, bsa_intra_unit_shape_config, sub_bsa_config, \
                key_suffixes=bsa_comp_key_suffixes, enable_suf_map=bsa_is_dense(da_config.bsa_config))# [NOTE]: Select best key_suffix from key_suffixes
        else:
            comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].\
                get_comp_map_key(schedule.da_config, [len(ids_tuple[0]), len(ids_tuple[1])], schedule.split_degrees)
        
        # Collect comp kernels to be fused
        old_kernels = []
        b_id = h_id = 0
        gpuid = None
        for xy in ids_set:
            x = xy[0]
            y = xy[1]
            if gpuid is None:
                gpuid = schedule.schedule_table[b_id, h_id, x, y]
            else:
                assert gpuid == schedule.schedule_table[b_id, h_id, x, y]
            comp_key = (b_id, h_id, x, y, gpuid)
            old_kernels.append(d_graph.kernel_dict[comp_key])
        
        # Create new comp kernel in d_graph.kernel_dict
        comp_key = (b_id, h_id, tuple(ids_tuple[0]), tuple(ids_tuple[1]), gpuid)
        new_kernel = Comp_Kernel(comp_key, schedule.m_config, comp_map_key, hierarchy)
        d_graph.kernel_dict[comp_key] = new_kernel
        
        # Update precursors and successors for all kernels
        for old_kernel in old_kernels:
            # for new kernel
            new_kernel.precursors.update(old_kernel.precursors)
            new_kernel.successors.update(old_kernel.successors)
            # for Comm kernels
            for precursor in old_kernel.precursors:
                precursor.successors.remove(old_kernel)
                precursor.add_successor(new_kernel)
            for successor in old_kernel.successors:
                successor.precursors.remove(old_kernel)
                successor.add_precursor(new_kernel)
            del d_graph.kernel_dict[old_kernel.key]
        
class Comp_Fusion_Substitution(Graph_Substitution):
    def  __init__(self, shape: tuple) -> None:
        super().__init__()
        assert len(shape) == 2
        self.shape = shape
    
    def dfs_lines(self, x_id: int, x_ids: list, y_ids: np.ndarray):
        if len(x_ids) == self.shape[0]:
            # Select every group of self.shape[1] y_ids in y_set and add (x_list, y_list) to self.cur_trans
            for selected_y_ids in itertools.combinations(y_ids, self.shape[1]):
                self.cur_trans.append(Comp_Fusion_Transformation(self, (np.array(x_ids), np.array(selected_y_ids))))
            return
        if x_id >= self.bool_schedule_table.shape[0]:
            return
        # not select x_id
        self.dfs_lines(x_id + 1, x_ids, y_ids)
        
        cur_y_ids = np.where(self.bool_schedule_table[x_id])[0] # np.ndarray
        new_y_ids = np.intersect1d(y_ids, cur_y_ids)
        if len(new_y_ids) < self.shape[1]:
            return
        # select x_id
        x_ids.append(x_id)
        self.dfs_lines(x_id + 1, x_ids, new_y_ids)
        x_ids.pop()
        
    def findall_trans_in_d_graph(self, d_graph: Dependent_Graph, hierarchy_sp: int) -> list:
        schedule_table = d_graph.schedule.schedule_table
        split_degrees = d_graph.split_degrees
        assert split_degrees[2] == split_degrees[3] == 1, "Not support bs_split or Nh_split > 1 !!!"
        schedule_table_sp = schedule_table[0, 0]    # (split_degrees[0], split_degrees[1]) <==> (Sq_split, Skv_split)
        assert schedule_table_sp.shape == tuple(split_degrees[: 2]), f"Error: {schedule_table_sp.shape} != {split_degrees[: 2]}"
        trans = []
        causal = False
        
        # Assign invalid values to diagonal of schedule table when causal !!!
        if schedule_table_sp[0, split_degrees[1] - 1] < 0: # causal
            causal = True
            assert split_degrees[0] == split_degrees[1]
            table_diagonal = copy.deepcopy(np.diagonal(schedule_table_sp))
            schedule_table_sp[np.diag_indices_from(schedule_table_sp)] = - 1
            
        for _ in range(hierarchy_sp):
            trans.append([])
        # Enumerate the top line of transformations
        
        for sp_id in range(hierarchy_sp):
            self.bool_schedule_table = schedule_table_sp == sp_id
            self.cur_trans = trans[sp_id]
            self.dfs_lines(0, [], np.arange(self.bool_schedule_table.shape[1]))
        
        # Retore diagonal values of schedule table
        if causal: # causal
            schedule_table_sp[np.diag_indices_from(schedule_table_sp)] = table_diagonal
        
        return trans
        
class Graph_Transformation_Engine():    # batch engine
    # input: d_graph
    # output: d_graph
    # [NOTE]: each position in schedule table cannot be fused more than ones !!!
    def __init__(self, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, m_config: Machine_Config):
        self.exp_config = exp_config
        self.da_config = da_config
        self.hierarchy = da_config.hierarchy
        self.tot_sp = da_config.tot_sp
        self.hierarchy_sp = da_config.SP[self.hierarchy]
    
    def print_trans(self):
        print(f'All transformations:', flush=True)
        for sp_id in range(self.hierarchy_sp):
            print(f'{sp_id}:', flush=True)
            for tran in self.trans_sp[sp_id]:
                print(f'{tran.ids_tuple}', flush=True)
    
    def get_all_transformations(self):
        '''
        transformations are concrete substitutions with positions on a concrete graph
        '''
        assert hasattr(self, 'd_graph'), 'No d_graph assigned !!!'
        d_graph = self.d_graph
        self.trans_sp = [] # different comp modules
        for _ in range(self.hierarchy_sp):
            self.trans_sp.append([])
        for sub in self.subs_dict['comp_fusion']:
            sub_trans = sub.findall_trans_in_d_graph(d_graph, self.hierarchy_sp)
            for sp_id in range(self.hierarchy_sp):
                self.trans_sp[sp_id] += sub_trans[sp_id]    # (sub0, tran0), (sub0, tran1), (sub1, tran0), (sub1, tran1), ...
        self.trans_all = []
        for trans in self.trans_sp:
            self.trans_all += trans
        # self.print_trans()
    
    def apply_transformations(self, selected_trans: list):
        new_d_graph = copy.deepcopy(self.d_graph)   # apply transformations on new_d_graph
        # Print trans
        print(f'Selected Transformations: ', end='', flush=True)
        for tran in selected_trans:
            print(f'{tuple(tran.ids_tuple)} ', end='', flush=True)
        print(flush=True)
        # Apply transformations on d_graph
        for tran in selected_trans:
            tran.apply_on_d_graph(new_d_graph)
        # Assess performance of new d_graph
        execute_plan = Execution_Plan(new_d_graph, self.exp_config.fob, plan_type=self.plan_type)
        # execute_plan.print_lp_result()
        return execute_plan
    
    def dfs_trans(self, trans_all_id: int, selected_trans: list, fused_pos: set):
        if trans_all_id >= len(self.trans_all):
            if len(selected_trans) == 0:
                return
            self.apply_transformations(selected_trans)
            return
        # not select
        self.dfs_trans(trans_all_id + 1, selected_trans, fused_pos)
        
        # select
        if len(fused_pos & self.trans_all[trans_all_id].ids_set) == 0:   # not conflict
            new_fused_pos = fused_pos | self.trans_all[trans_all_id].ids_set
            selected_trans.append(self.trans_all[trans_all_id])
            self.dfs_trans(trans_all_id + 1, selected_trans, new_fused_pos)
            selected_trans.pop()

    def calc_subs_dict(self):
        # type1: comp fusion substitutions
        # print(f'self.da_config.bsa_config: {self.da_config.bsa_config}', flush=True)
        # if self.da_config.bsa_config is None: # dense
        if bsa_is_dense(self.da_config.bsa_config): # dense: just causal !!!
            # self.comp_unit_ub = self.hierarchy_sp // 2 + (self.hierarchy_sp == 3)  # 4 -> 2, 5 -> 2, 8 -> 4, special !!! 3 -> 2
            assert self.d_graph.split_degrees[0] == self.d_graph.split_degrees[1]
            Par_D = self.d_graph.split_degrees[0]
            self.comp_unit_ub = int(math.ceil(Par_D * (Par_D - 1) / 2 / self.hierarchy_sp))
        else:   # bsa
            self.comp_unit_ub = int(math.ceil(math.prod(self.d_graph.split_degrees) / self.hierarchy_sp))
        # self.ub_factors = get_factors(self.comp_unit_ub)
        self.comp_fusion_shapes = []
        for x in range(1, int(self.comp_unit_ub ** 0.5) + 1):
            for y in range(x, self.comp_unit_ub // x + 1):
                if x == 1 and y == 1:
                    continue
                if x * y <= self.comp_unit_ub:
                    self.comp_fusion_shapes.append((x, y))
                if x != y:
                    self.comp_fusion_shapes.append((y, x))
        self.comp_fusion_shapes.sort(key=lambda x: (x[0] * x[1], x[1]), reverse=True)
        # print_rank_0(f'self.comp_unit_ub: {self.comp_unit_ub}')
        # print_rank_0(f'comp_fusion_shapes: {self.comp_fusion_shapes}')
        self.comp_fusion_subs = [Comp_Fusion_Substitution(shape) for shape in self.comp_fusion_shapes]
                
        # type2: comm fusion substitutions [TODO]
        self.comm_fusion_subs = []
        
        # sort all substitutions by performance
        self.comp_fusion_subs.sort(key=lambda x: (math.prod(x.shape), x.shape), reverse=True)

        # substitutions dictionary
        self.subs_dict = {
            'comp_fusion': self.comp_fusion_subs,
            'comm_fusion': self.comm_fusion_subs,
        }
        # for sub in self.subs_dict['comp_fusion']:
        #     print(f'sub.shape: {sub.shape}', flush=True)
        
    def transform(self, d_graph: Dependent_Graph, mode: str = 'bf', plan_type: str = 'automatic'):
        self.d_graph = d_graph
        self.plan_type = plan_type
        self.calc_subs_dict()
        self.get_all_transformations()
        if mode == 'bf':
            self.dfs_trans(0, [], set())
        elif mode == 'greedy':
            # Select transformations greedly
            fused_pos = set()
            selected_trans = []
            for tran in self.trans_all:
                if len(fused_pos & tran.ids_set) == 0: # select this trans
                    fused_pos |= tran.ids_set
                    selected_trans.append(tran)
            # print selected trans
            if len(selected_trans) == 0:
                print(f'No Transformations Selected !!!', flush=True)
                # return None
            execute_plan = self.apply_transformations(selected_trans)
            return execute_plan
        else:
            raise Exception(f'Error: mode {mode} not supported !!!')
