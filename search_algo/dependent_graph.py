import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, FlashAttn_Profile_Map, Machine_Config
from search_algo.bsa_utils import convert_shape_config_to_str, bsa_is_dense, bsa_is_full, bsa_is_causal, get_b2f_suf_map, \
    select_best_profile_comp_key, bsa_repr_is_causal, bsa_repr_is_square_full
from search_algo.bsa_config import BSA_Config
from search_algo.search_engine import Dist_Attn_Config
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
        return self.is_empty_kernel[fob] if hasattr(self, 'is_empty_kernel') else self.time[fob] <= 0
    
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
                 time: Optional[np.array] = None, seqlen_variable_graph = False):
        '''
        comp_map_key is useless when seqlen_variable_graph = True
        '''
        # dict keys: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids), (c_ids), gpuid)
        super().__init__(key, 'comp')
        # kernel time
        # if hierarchy == 0 and time is None:  # Only print at inter level without hack !!!
        #     print_rank_0(f'comp_map_key: {comp_map_key}; time: {time}, seqlen_variable_graph: {seqlen_variable_graph}')
        if not seqlen_variable_graph:
            if time is None:
                # flashattn profile map_key:
                self.comp_map_key = comp_map_key    # profile_comp_map_key at inter level, flashattn_profile_map_key at intra level
                # one of them is None for inter_bsa !!!
                self.time = m_config.comp_profile_maps[hierarchy].get_comp_time_from_map_key(comp_map_key) # [fwd/bwd];
            else:
                self.time = time
        else:
            self.is_empty_kernel = np.array([False, False], dtype=np.bool)  # Comp kernels should be empty kernel
            self.time = np.array([1e-3, 1e-2])  # dummy duration times

class Comm_Kernel(Cuda_Kernel):
    def __init__(self, key: tuple, m_config: Machine_Config, comm_raw_map_key: tuple, units: np.ndarray, hierarchy: int, seqlen_variable_graph=False):
        # dict keys: (b_id, h_id, r/c_id, send, recv, i/o, r/c)
        super().__init__(key, 'comm')
        # Bytes of data to send/recv
        if not seqlen_variable_graph:
            self.comp_raw_map_key =  comm_raw_map_key
        assert units.shape == (2,)
        self.units = units  # [fwd/bwd]
        self.hierarchy = hierarchy
        # kernel time
        if not seqlen_variable_graph:
            self.time = np.array([
                m_config.comm_profile_maps[hierarchy].get_comm_time_from_map_key((comm_raw_map_key[0] * unit,))
                for unit in units
            ])    # [fwd/bwd]
        else:
            self.is_empty_kernel = units == 0
            self.time = np.array([1e-3, 1e-2])  # dummy duration times

class Dependent_Graph():
    def __init__(self, schedule: Dist_Attn_Schedule, fob: bool, kernel_dict: Optional[dict] = None, is_inter_bsa = False, \
                    bsa_comp_key_suffixes: List[str] = None, seqlen_variable_graph = False):
        """_summary_

        Args:
            schedule (Dist_Attn_Schedule): _description_
            fob (bool): _description_
            kernel_dict (Optional[dict], optional): _description_. Defaults to None.
            is_inter_bsa (bool, optional): _description_. Defaults to False.
            seqlen_variable_graph (bool, optimal): If set True, the seqlen of this dependent graph can be variable which meas \
                execution time and map_key for each kernel is meaningless. Currently, if can be True only when the pattern is full.
            # only_graph_structure (bool, optional): If set True, this dependent graph only describes structure of the graph without \
            #     real execution time for each kernel. Defaults to False. Currently, if can be True only when the pattern is full.
        """
        # [NOTE]: only support star tree of broadcase/reduce here !!!
        # build dependent graph from schedule_table
        
        self.schedule = schedule
        # self.da_config = schedule.da_config
        # self.m_config = schedule.m_config
        # self.split_degrees = schedule.split_degrees
        self.fob = fob  # fwd or bwd
        # self.tot_sp = schedule.tot_sp
        # self.hierarchy = schedule.da_config.hierarchy
        # self.is_inter_bsa = is_inter_bsa    # [DDPRECATED]
        self.bsa_comp_key_suffixes = bsa_comp_key_suffixes
        self.seqlen_variable_graph = seqlen_variable_graph  # [TODO]: convert `seqlen_variable_graph` to `shape_variable_graph`
        # self.root_kernel = Cuda_Kernel()
        # comp: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids), (c_ids), gpuid) -> Cuda_Kernel
        # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
        self.kernel_dict = {}
        # Build kernel_dict
        if not kernel_dict:
            self.create_raw_graph() # generate kernel_dict
        else:
            self.kernel_dict = kernel_dict
            
    # Deducible attributes
    @property
    def da_config(self) -> Dist_Attn_Config:
        return self.schedule.da_config
    
    @property
    def m_config(self):
        return self.schedule.m_config
    
    @property
    def split_degrees(self):
        return self.schedule.split_degrees
    
    @property
    def tot_sp(self):
        return self.schedule.tot_sp
    
    @property
    def hierarchy(self):    # (0, 1) -> (inter, intra)
        return self.schedule.da_config.hierarchy
    
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        new_self = copy.copy(self)
        new_self.kernel_dict = copy.deepcopy(self.kernel_dict, memo)
        return new_self
    
    def select_bsa_comp_key(self, CP: tuple, shape_config: dict, bsa_config: BSA_Config, key_suffixes: List[str], enable_suf_map=False):
        '''
        Return: profile_comp_map_key at intra level
        '''
        fob = self.fob
        hierarchy = self.hierarchy
        # print_rank_0(f'convert_shape_config_to_str(shape_config): {convert_shape_config_to_str(shape_config)}')
        intra_CP = CP[0]
        if enable_suf_map and bsa_is_full(bsa_config):  # use suffix map when inter pattern is causal and intra pattern is full
            profile_comp_key_suffixes = list(set([suf for key_suffix in key_suffixes for suf in get_b2f_suf_map(intra_CP)[key_suffix]]))
            # [HACK]: Manually create a full bsa_config when Sq != Skv
            bsa_config = BSA_Config.create_full(CP)
        else:
            profile_comp_key_suffixes = key_suffixes
        key_preffix = f'fob={fob}_CP={CP}_shape_config={{{convert_shape_config_to_str(shape_config)}}}_bsa_config={{{bsa_config}}}'
        
        def fault_tolerance_func(key_suffix: str):
            # When Nh=32 and intra_bsa=full and inter_bsa is dense, 'w_kernel_tile' does not exist !!!
            return enable_suf_map and bsa_is_full(bsa_config) and max(shape_config['Nh']) > 1 and 'w_kernel_tile' in key_suffix
        
        best_profile_comp_key = select_best_profile_comp_key(key_preffix, profile_comp_key_suffixes, self.m_config.comp_profile_maps[hierarchy], \
            fob, fault_tolerance_func)
        return best_profile_comp_key
        # opt_comp_time = float('inf')
        # for key_suffix in profile_comp_key_suffixes:
        #     comp_map_key = f'{key_preffix}{key_suffix}'
        #     try:
        #         cur_comp_time = self.m_config.comp_profile_maps[hierarchy].get_comp_time_from_map_key(comp_map_key)[fob]
        #     except Exception as e:
        #         if not (enable_suf_map and bsa_is_full(bsa_config) and max(shape_config['Nh']) > 1 and 'w_kernel_tile' in key_suffix):
        #             raise e
        #     if cur_comp_time < opt_comp_time:
        #         selected_key_suffix = key_suffix
        #         opt_comp_time = cur_comp_time
        # return f'{key_preffix}{selected_key_suffix}'
    
    def is_materialized(self):
        return self.seqlen_variable_graph is False
    
    def materialize(self, da_config: Dist_Attn_Config):
        assert self.seqlen_variable_graph is True, f'Now only support materializing Execution_Plan with seqlen_variable_graph'
        assert set(self.da_config.get_distinction_kv_dict(da_config).keys()).issubset(set(('Nh', 'S', 'bs', 'D'))), \
            f'Now only support varying shape_config in dependenct graph'
        # Update da_config & seqlen_variable_graph
        self.schedule.update_da_config(da_config)
        self.seqlen_variable_graph = False
        # Rebuild raw_graph
        self.create_raw_graph()
    
    def create_raw_graph(self):
        '''
        OBJ: Clear old graph and rebuild raw graph through rebuilding self.kernel_dict
        '''
        # clear old kernel_dict
        self.kernel_dict = {}
        # rebuild raw graph
        schedule = self.schedule
        hierarchy = self.hierarchy  # (0, 1) -> (inter, intra)
        fob = self.fob
        da_config = self.da_config
        # step1: Build Comp Kernel
        # if self.is_inter_bsa:
        if not self.seqlen_variable_graph:
            if self.hierarchy == 0: # inter
                intra_CP_tuple = (self.da_config.SP[1], 1)  # (intra, inter)
                inter_CP = self.da_config.SP[0]
                if not bsa_is_dense(da_config.bsa_config):  # Only causal allows inter_CP < Par_D
                    assert inter_CP == schedule.split_degrees[0] == schedule.split_degrees[1]
                bsa_intra_unit_shape_config = copy.deepcopy(self.da_config.shape_config)
                # [NOTE]: Seqlen for smallest scheduling unit !!!
                bsa_intra_unit_shape_config['S'] = (bsa_intra_unit_shape_config['S'][0] // schedule.split_degrees[0], \
                                                    bsa_intra_unit_shape_config['S'][1] // schedule.split_degrees[1])
            else:   # intra (full or intra_bsa❌)
                comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
                causal_comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [1, 1], schedule.split_degrees, causal=True)
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                for k in range(schedule.split_degrees[0]):   # split_Sq, may > CP
                    for l in range(schedule.split_degrees[1]):   # split_Skv, may > CP
                        if schedule.schedule_table[i, j, k, l] >= 0:    # Valid comp kernel
                            comp_key = (i, j, k, l, schedule.schedule_table[i, j, k, l])
                            assert comp_key not in self.kernel_dict.keys()
                            # if self.is_inter_bsa:   # ✅
                            if self.seqlen_variable_graph:
                                profile_comp_map_key = None
                            else:
                                sub_bsa_repr = da_config.bsa_config.bsa_repr.create_sub_bsa_repr(
                                    schedule.split_degrees[0: 2], select_ids = [[k], [l]])
                                if self.hierarchy == 0: # inter # [TODO]
                                    sub_bsa_config = BSA_Config(None, None, {'bsa_repr': sub_bsa_repr, 'CP': intra_CP_tuple})
                                    profile_comp_map_key = self.select_bsa_comp_key(intra_CP_tuple, bsa_intra_unit_shape_config, sub_bsa_config, \
                                        key_suffixes=self.bsa_comp_key_suffixes, enable_suf_map=bsa_is_dense(da_config.bsa_config))# [NOTE]: Select best key_suffix from key_suffixes
                                else:
                                    if bsa_repr_is_causal(sub_bsa_repr):
                                        profile_comp_map_key = causal_comp_map_key
                                    elif bsa_repr_is_square_full(sub_bsa_repr):
                                        profile_comp_map_key = comp_map_key
                                    else:
                                        raise Exception(f'[ERROR]: Unknown sub_bsa_repr={sub_bsa_repr}')
                            self.kernel_dict[comp_key] = Comp_Kernel(
                                comp_key, schedule.m_config,
                                profile_comp_map_key,
                                hierarchy,
                                seqlen_variable_graph=self.seqlen_variable_graph)
        # step2: Build Comm Kernel
        assert schedule.split_degrees[0] == schedule.split_degrees[1] # [NOTE]: now only support Sq_split == Skv_split !!!
        comm_raw_map_key = None if self.seqlen_variable_graph else \
            schedule.m_config.comm_profile_maps[hierarchy].get_comm_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
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
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_row, hierarchy,
                                                                     seqlen_variable_graph=self.seqlen_variable_graph)
                            # output row reduce
                            comm_key = (i, j, k, dst_g_id, cur_g_id, 'o', 'r')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_row, hierarchy,
                                                                     seqlen_variable_graph=self.seqlen_variable_graph)
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
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_col, hierarchy,
                                                                     seqlen_variable_graph=self.seqlen_variable_graph)
                            # output col reduce
                            comm_key = (i, j, l, dst_g_id, cur_g_id, 'o', 'c')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_col, hierarchy,
                                                                     seqlen_variable_graph=self.seqlen_variable_graph)
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
