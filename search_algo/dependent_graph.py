import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, FlashAttn_Profile_Map, Machine_Config
from search_algo.bsa_utils import convert_shape_config_to_str
from search_algo.bsa_config import BSA_Config
import numpy as np
from typing import Optional, List
import copy
from search_algo.utils import print_rank_0

class Cuda_Kernel():
    def __init__(self, key: tuple, type: str):
        self.key = key
        self.type = type
        self.precursors = set()
        self.successors = set()
    
    def add_precursor(self, precursor):
        self.precursors.add(precursor)
    
    def add_successor(self, successor):
        self.successors.add(successor)
    
    def remove_precursor(self, precursor):
        self.precursors.discard(precursor)
    
    def remove_successor(self, successor):
        self.successors.discard(successor)
    
    def is_empty(self, fob):
        return self.time[fob] <= 0
    
    def add_edge(self, v, fob):
        # check whether self or v is empty
        if self.is_empty(fob) or v.is_empty(fob):
            return
        self.add_successor(v)
        v.add_precursor(self)
    
    def remove_edge(self, v, fob):
        # check whether self or v is empty
        if self.is_empty(fob) or v.is_empty(fob):
            return
        self.remove_successor(v)
        v.remove_precursor(self)
                
class Comp_Kernel(Cuda_Kernel):
    def __init__(self, key: tuple, m_config: Machine_Config, comp_map_key: tuple, hierarchy: int, \
                 time: Optional[np.array] = None):
        # dict keys: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids), (c_ids), gpuid)
        super().__init__(key, 'comp')
        # kernel time
        if hierarchy == 0 and time is None:  # Only print at inter level without hack !!!
            print_rank_0(f'comp_map_key: {comp_map_key}; time: {time}')
        if time is None:
            # flashattn profile map_key:
            self.comp_map_key = comp_map_key
            # one of them is None for inter_bsa !!!
            self.time = m_config.comp_profile_maps[hierarchy].get_comp_time_from_map_key(comp_map_key) # [fwd/bwd];
        else:
            self.time = time
    
    
class Comm_Kernel(Cuda_Kernel):
    def __init__(self, key: tuple, m_config: Machine_Config, comm_raw_map_key: tuple, units: np.ndarray, hierarchy: int):
        # dict keys: (b_id, h_id, r/c_id, send, recv, i/o, r/c)
        super().__init__(key, 'comm')
        # Bytes of data to send/recv
        self.comp_raw_map_key = comm_raw_map_key
        assert units.shape == (2,)
        self.units = units  # [fwd/bwd]
        self.hierarchy = hierarchy
        # kernel time
        self.time = np.array([
            m_config.comm_profile_maps[hierarchy].get_comm_time_from_map_key((comm_raw_map_key[0] * unit,))
            for unit in units
        ])    # [fwd/bwd]

class Dependent_Graph():
    def __init__(self, schedule: Dist_Attn_Schedule, fob: bool, kernel_dict: Optional[dict] = None, is_inter_bsa = False, \
                    bsa_comp_key_suffixes: List[str] = None):
        # [NOTE]: only support star tree of broadcase/reduce here !!!
        # build dependent graph from schedule_table
        
        self.schedule = schedule
        self.da_config = schedule.da_config
        self.m_config = schedule.m_config
        self.split_degrees = schedule.split_degrees
        self.fob = fob  # fwd or bwd
        self.tot_sp = schedule.tot_sp
        self.hierarchy = schedule.da_config.hierarchy
        self.is_inter_bsa = is_inter_bsa
        self.bsa_comp_key_suffixes = bsa_comp_key_suffixes
        # self.root_kernel = Cuda_Kernel()
        # comp: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids), (c_ids), gpuid) -> Cuda_Kernel
        # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
        self.kernel_dict = {}
        if not kernel_dict:
            self.create_raw_graph()
        else:
            self.kernel_dict = kernel_dict
    
    # @classmethod
    # def create_from_other_d_graph(cls, other_d_graph):
    #     cls(other_d_graph.schedule, other_d_graph.fob, other_d_graph.kernel_dict)
    
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        new_self = copy.copy(self)
        new_self.kernel_dict = copy.deepcopy(self.kernel_dict, memo)
        return new_self
    
    def select_bsa_comp_key(self, CP: tuple, shape_config: dict, bsa_config: BSA_Config, key_suffixes: List[str]):
        fob = self.fob
        hierarchy = self.hierarchy
        key_preffix = f'fob={fob}_CP={CP}_shape_config={{{convert_shape_config_to_str(shape_config)}}}_bsa_config={{{bsa_config}}}'
        opt_comp_time = float('inf')
        for key_suffix in key_suffixes:
            comp_map_key = f'{key_preffix}{key_suffix}'
            cur_comp_time = self.m_config.comp_profile_maps[hierarchy].get_comp_time_from_map_key(comp_map_key)[fob]
            if cur_comp_time < opt_comp_time:
                selected_key_suffix = key_suffix
                opt_comp_time = cur_comp_time
        return f'{key_preffix}{selected_key_suffix}'
        
    def create_raw_graph(self):
        schedule = self.schedule
        hierarchy = self.hierarchy  # (0, 1) -> (inter, intra)
        fob = self.fob
        da_config = self.da_config
        # step1: Build Comp Kernel
        if self.is_inter_bsa:
            assert hierarchy == 0
            bsa_CP = (self.da_config.SP[1], 1)  # (intra, inter)
            inter_CP = self.da_config.SP[0]
            bsa_intra_shape_config = copy.deepcopy(self.da_config.shape_config)
            bsa_intra_shape_config['S'] = (bsa_intra_shape_config['S'][0] // inter_CP, bsa_intra_shape_config['S'][1] // inter_CP)
            # bsa_comp_key: {fob, CP, shape_config, bsa_config}
        else:   # dense or inter_bsa
            comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
            causal_comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [1, 1], schedule.split_degrees, causal=True)
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        if schedule.schedule_table[i, j, k, l] >= 0:    # Valid comp kernel
                            comp_key = (i, j, k, l, schedule.schedule_table[i, j, k, l])
                            assert comp_key not in self.kernel_dict.keys()
                            if self.is_inter_bsa:   # ✅
                                sub_bsa_repr = da_config.bsa_config.bsa_repr.create_sub_bsa_repr(
                                    schedule.split_degrees[0: 2], select_ids = [[k], [l]])
                                sub_pat_bsa_repr = {
                                    'bsa_repr': sub_bsa_repr,
                                    'CP': bsa_CP,
                                }
                                sub_bsa_config = BSA_Config(None, None, sub_pat_bsa_repr)
                                real_comp_map_key = self.select_bsa_comp_key(bsa_CP, bsa_intra_shape_config, sub_bsa_config, \
                                    key_suffixes=self.bsa_comp_key_suffixes)   # [NOTE]: Select best key_suffix from key_suffixes
                            else:
                                # bsa should set da_config.causal too !!!✅
                                real_comp_map_key = causal_comp_map_key if da_config.causal and k == l else comp_map_key
                            self.kernel_dict[comp_key] = Comp_Kernel(
                                comp_key, schedule.m_config,
                                real_comp_map_key,
                                hierarchy)
        # step2: Build Comm Kernel
        assert schedule.split_degrees[0] == schedule.split_degrees[1] # [NOTE]: now only support Sq_split == Skv_split !!!
        comm_raw_map_key = schedule.m_config.comm_profile_maps[hierarchy].get_comm_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                # row
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    cur_g_id = schedule.S_map[i, k]
                    dst_set = set()
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0 and dst_g_id != cur_g_id and dst_g_id not in dst_set:
                            dst_set.add(dst_g_id)
                            # input row broadcast
                            comm_key = (i, j, k, cur_g_id, dst_g_id, 'i', 'r')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_row, hierarchy)
                            # output row reduce
                            comm_key = (i, j, k, dst_g_id, cur_g_id, 'o', 'r')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_row, hierarchy)
                # col
                for l in range(schedule.split_degrees[1]):  # split_Skv
                    cur_g_id = schedule.S_map[j, l]
                    dst_set = set()
                    for k in range(schedule.split_degrees[0]):  # split_Sq
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0 and dst_g_id != cur_g_id and dst_g_id not in dst_set:
                            dst_set.add(dst_g_id)
                            # input col broadcast
                            comm_key = (i, j, l, cur_g_id, dst_g_id, 'i', 'c')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_col, hierarchy)
                            # output col reduce
                            comm_key = (i, j, l, dst_g_id, cur_g_id, 'o', 'c')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_col, hierarchy)
        # [NOTE]: every nonempty kernel in self.kernel_dict should be launched by Execute_Engine
        
        # step3: Build dependences between kernels, differentiate fwd and bwd !!!
        # comp kernel centric
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0:    # Valid comp kernel
                            comp_key = (i, j, k, l, dst_g_id)
                            comp_kernel = self.kernel_dict[comp_key]
                            cur_g_id = schedule.S_map[i, k]
                            if dst_g_id != cur_g_id:
                                # input row broadcast
                                comm_key = (i, j, k, cur_g_id, dst_g_id, 'i', 'r')
                                self.kernel_dict[comm_key].add_edge(comp_kernel, fob)
                                # output row reduce
                                comm_key = (i, j, k, dst_g_id, cur_g_id, 'o', 'r')
                                comp_kernel.add_edge(self.kernel_dict[comm_key], fob)
                            
                            cur_g_id = schedule.S_map[i, l]
                            if dst_g_id != cur_g_id:
                                # input col broadcast
                                comm_key = (i, j, l, cur_g_id, dst_g_id, 'i', 'c')
                                self.kernel_dict[comm_key].add_edge(comp_kernel, fob)
                                # output col reduce
                                comm_key = (i, j, l, dst_g_id, cur_g_id, 'o', 'c')
                                comp_kernel.add_edge(self.kernel_dict[comm_key], fob)
    
             
        
