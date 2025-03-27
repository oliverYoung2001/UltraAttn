
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from queue import PriorityQueue
from heapq import heappush, heappop, heappushpop
from search_algo.global_vars import *
import math
import regex as re
import numpy as np
from typing import Optional, List
import torch
from fractions import Fraction
import inspect
import argparse
import regex as re

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def find_file_with_regex(dir: str, pat: str) -> List[str]:
    matched_files = []
    for file_name in os.listdir(dir):
        if re.match(pat, file_name):
            matched_files.append(file_name)
    return matched_files

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--attn-mode", type=str, choices=['zigzag_ring', 'lightseq', 'local_flash'], default="flash")
    parser.add_argument('--profiler-with-tensorboard', action='store_true', default=False, help='whether to profile with tensorboard')
    parser.add_argument('--tb-dir', default=None, type=str, help='where to storage tensorboard files')

    args = parser.parse_args()
    return args

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def calc_flops(mbs, S: tuple, Nh, D, causal=True, fob=0, total_sparsity=1):
    # print(f'total_sparsity: {total_sparsity}')
    flops = 2 * 2 * mbs * S[0] * S[1] * Nh * D * total_sparsity
    if fob == 0:
        m_flops = h_flops = flops
    elif fob == 1:
        m_flops = 2 * flops
        h_flops = 2.5 * flops
    elif fob == 2:
        m_flops = (1 + 2) * flops
        h_flops = (1 + 2.5) * flops
    return m_flops, h_flops # model flops & hardware flops

def all_wait_main_stream(stream_list: list, main_stream: torch.cuda.Stream):
    for stream in stream_list:
        if stream.cuda_stream != main_stream.cuda_stream:
            stream.wait_stream(main_stream)

def main_stream_wait_all(stream_list: list, main_stream: torch.cuda.Stream):
    for stream in stream_list:
        if stream.cuda_stream != main_stream.cuda_stream:
            main_stream.wait_stream(stream)

def convert_block_table_to_value(block_table):
    block_table_value = np.array([v.value for v in block_table.flatten()]).reshape(block_table.shape)
    return block_table_value
        
def combine_list_to_0(l0, l1):
    for x in l1:
        if x not in l0:
            l0.append(x)
    return l0

def unique_list(l):
    u_l = []
    for x in l:
        if x not in u_l:
            u_l.append(x)
    return u_l

def closest_fraction(x, max_denominator=1000):
    return Fraction(x).limit_denominator(max_denominator)

class Block_Table_Type(Enum):
    EMPTY = 0
    FULL = 1
    PARTIAL = 2
    
class Block_Type(Enum):
    EMPTY = 0
    FULL = 1
    CAUSAL = 2

Block_Comp_Volume = {
    Block_Type.EMPTY: 0,
    Block_Type.FULL: 1,
    Block_Type.CAUSAL: 0.5,
}

class Block_Attention_Config():
    CP: int  # degree of CP
    ParD: int  # degree of partition
    cmap: np.ndarray  # Contxet map: context chunk -> CP_rank
    block_table: np.ndarray  # 0 -> empty, 1 -> full, 2-> causal
    
    def __init__(self, CP: int, ParD: int, cmap: np.ndarray, block_table: np.ndarray):
        self.CP = CP
        self.ParD = ParD
        self.cmap = cmap
        self.block_table = block_table
        block_table_value = convert_block_table_to_value(block_table)
        # block_table_value = np.array([v.value for v in block_table.flatten()]).reshape(block_table.shape)
        print(f'block_table_value:\n{block_table_value}')
    
    @classmethod
    def from_causal(cls, CP: int, ParD: int, cmap: np.ndarray):
        block_table = np.zeros((ParD, ParD), dtype=Block_Type)
        # clear block_table to empty
        for i in range(ParD):
            for j in range(ParD):
                block_table[i, j] = Block_Type.EMPTY
        # set block_table to causal
        for i in range(ParD):
            for j in range(i):
                block_table[i, j] = Block_Type.FULL
            block_table[i, i] = Block_Type.CAUSAL
        return cls(CP, ParD, cmap, block_table)
    
    # @classmethod
    # def from_custom(cls, CP: int, ParD: int, cmap: np.ndarray, block_table: np.ndarray):
    #     return cls(CP, ParD, cmap, block_table)
    
def calc_table_comp_relative_time(block_table: np.ndarray):
    return sum([Block_Comp_Volume[block_table[i, j]] for i in range(block_table.shape[0]) for j in range(block_table.shape[1])])

def get_block_table_type(block_table: np.ndarray) -> Block_Table_Type:
    IS_EMPTY = True
    IS_FULL = True
    for i in range(block_table.shape[0]):
        for j in range(block_table.shape[1]):
            if block_table[i, j].value != Block_Type.EMPTY.value:
                IS_EMPTY = False
            if block_table[i, j].value != Block_Type.FULL.value:
                IS_FULL = False
    return Block_Table_Type.EMPTY if IS_EMPTY else (Block_Table_Type.FULL if IS_FULL else Block_Table_Type.PARTIAL)
    
def get_factors(n: int):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

class FinitePriorityQueue():
    """finite min heap
    """
    def __init__(self, maxsize: int, pack_func, unpack_func):
        self.maxsize = maxsize
        self.queue = []
        self.pack_func = pack_func
        self.unpack_func = unpack_func
    
    def reset(self):
        self.queue = []

    def push(self, item):
        if len(self.queue) < self.maxsize:
            heappush(self.queue, self.pack_func(item))
        else:
            heappushpop(self.queue, self.pack_func(item))

    def pop(self):
        return self.unpack_func(heappop(self.queue))

    def __len__(self):
        return len(self.queue)
    
def convert_profile_data_to_map(profile_list):
    profile_map = {}
    for i in range(len(profile_list)):
        map_key = tuple(profile_list[i][0])
        assert map_key not in profile_map.keys()
        profile_map[map_key] = np.array(profile_list[i][1][:2]) / 1e6    # [fwd/bwd], (s)
    # print(f'profile_map: {profile_map}')
    return profile_map

# # Helper function to pretty-print message sizes
# def convert_throughput(size_bytes, round_=3):
#     if size_bytes == 0:
#         return "0B"
#     size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
#     i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_DOWN)))
#     p = math.pow(BYTE_MULTPLE_DOWN, i)
#     s = round(size_bytes / p, round_)
#     return "%s %s" % (s, size_name[i])

def convert_throughput_to_B(size: float, unit: str):
    size_name = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    assert unit in size_name, f'Invalid unit: {unit}'
    return size * pow(BYTE_MULTPLE_DOWN, size_name.index(unit)) # ?B -> B
    
def convert_profile_data_to_comm_map(file_name: str, num_gpus_div: int):
    profile_map = {}    # Bytes -> GB/s
    # SIZE 8192, REAL_BD 403.402 MB/s, BD/PAIR 201.701 MB/s, time 0.0041 s, comm_vol 1.638 MB
    pat1 = re.compile(r'^SIZE (\d+),.*?BD/PAIR (\d*(\.\d*)?) ([A-Z]*)/s.*$')
    # SIZE 131072, REAL_BD 16.653 GB/s, time 0.0013 s, comm_vol 20.972 MB
    pat2 = re.compile(r'^SIZE (\d+),.*?REAL_BD (\d*(\.\d*)?) ([A-Z]*)/s.*$')

    with open(file_name, 'r') as f:
        for line in f.readlines():
            res = pat1.match(line)
            if res is None:
                res = pat2.match(line)
            if res is None:
                continue
            profile_map[(int(res.group(1)),)] = convert_throughput_to_B(float(res.group(2)), res.group(4)) \
                                                / pow(BYTE_MULTPLE_DOWN, 3) / num_gpus_div
    # print(f'profile_map: {profile_map}')
    return profile_map

def convert_node_profile_data_to_comp_map(file_name: Optional[str], local_size: int):
    # map_key: ((Sq, Skv), (Nhq, Nhg), bs, D, causal) -> Time[fwd/bwd]  # S per GPU !!!
    print_rank_0(f'node_profile_data file_name: {file_name}')
    if file_name is None:
        return {}
    profile_map = {
        'level0': {},
        'level1': {},
    }
    # fob = SP = S = Nh = bs = D = causal = None
    # cur_map_key = None
    
# fob=0, plan_paths: ['/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=1_X=8_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=2_X=4_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=4_X=2_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=8_X=1_dim=0.pkl']

# da_config: SP=(1,8),Sg=(256,256),S=(2048,2048),Nh=(1,1),bs=1,D=128,causal=False,hierarchy=1
# # orchestrated_attn_func
# mfu: 0.832 Tflops/s, hfu: 0.832 Tflops/s, 3100.198 iter/s, 3.226e-04 s/iter, (10.486, 0.002, 0.001) sec
# mfu: 1.256 Tflops/s, hfu: 1.256 Tflops/s, 4678.144 iter/s, 2.138e-04 s/iter, (13.294, 0.003, 0.001) sec
# mfu: 1.336 Tflops/s, hfu: 1.336 Tflops/s, 4976.115 iter/s, 2.010e-04 s/iter, (13.712, 0.002, 0.001) sec
# mfu: 1.364 Tflops/s, hfu: 1.364 Tflops/s, 5079.649 iter/s, 1.969e-04 s/iter, (14.117, 0.000, 0.001) sec
# # orchestrated_attn_func fused
# mfu: 2.608 Tflops/s, hfu: 2.608 Tflops/s, 9717.040 iter/s, 1.029e-04 s/iter, (15.720, 0.000, 0.000) sec
# mfu: 2.857 Tflops/s, hfu: 2.857 Tflops/s, 10643.733 iter/s, 9.395e-05 s/iter, (17.253, 0.001, 0.000) sec
# mfu: 3.616 Tflops/s, hfu: 3.616 Tflops/s, 13469.827 iter/s, 7.424e-05 s/iter, (18.530, 0.002, 0.000) sec
# mfu: 3.679 Tflops/s, hfu: 3.679 Tflops/s, 13706.140 iter/s, 7.296e-05 s/iter, (18.940, 0.002, 0.000) sec

    baselines = ['ring_flash_attn_func', 'zigzag_ring_flash_attn_func', 'stripe_flash_attn_func']
    skip_lines = 0
    pat0 = re.compile(r'^fob=(\d).*$')
    # pat1 = re.compile(r'^SP=\((\d+),(\d+)\),S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat1 = re.compile(r'^.*SP=\((\d+),(\d+)\),.*S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat2 = re.compile(r'^.*iter/s, (-?(\d+(?:\.\d+)?(?:e[+-]\d+)?)) s/iter,.*$')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if skip_lines > 0:
                skip_lines -= 1
                continue
            for baseline in baselines:  # [NOTE]: Results of baseline models are error !!!
                if baseline in line:
                   skip_lines = 1
                #    print(f'baseline: {baseline}')
                   continue 
            if 'orchestrated_attn_func' in line:
                cur_level = 0
                if 'fused' in line:
                    cur_level = 1
                continue
            res = pat2.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                assert cur_map_key is not None and fob is not None
                if cur_map_key not in profile_map['level0']:
                    for l in range(2):
                        profile_map[f'level{l}'][cur_map_key] = np.empty((2,), dtype=np.float32)
                        profile_map[f'level{l}'][cur_map_key].fill(np.inf)
                # print(f'cur_map_key: {cur_map_key}, fob: {fob}, cur_level: {cur_level}, {float(res.group(1))}')
                for l in range(cur_level, 2):
                    profile_map[f'level{l}'][cur_map_key][fob] = min(profile_map[f'level{l}'][cur_map_key][fob], float(res.group(1)))
                continue
            res = pat1.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}, {res.group(2)}, {res.group(3)}, {res.group(4)}, {res.group(5)}, {res.group(6)}, {res.group(7)}, {res.group(8)}, {res.group(9)}')
                SP = (int(res.group(1)), int(res.group(2)))
                tot_sp = SP[0] * SP[1]
                S = (int(res.group(3)) // tot_sp, int(res.group(4)) // tot_sp)
                # S = (int(res.group(3)), int(res.group(4)))  # just for analysis !!!
                Nh = (int(res.group(5)), int(res.group(6)))
                bs = int(res.group(7))
                D = int(res.group(8))
                causal = res.group(9) == 'True'
                cur_map_key = (SP, S, Nh, bs, D, causal)
                # print(f'pat1, fob={fob}, cur_map_key: {cur_map_key}')
                continue
            res = pat0.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                fob = int(res.group(1))
                # print(f'fob: {fob}')
                continue
            
    # print(f'{profile_map}')
    # For intra full attn:
    # 'level0': {1 out of 4}
    # 'level1': {1 out of all (8 for Nh=1, 4 for Nh=32)}
    # 'level2': {1 out of all ...}
    profile_map['level2'] = profile_map['level1']   # [NOTE]: Passed for correction
    # print(f'level0: {profile_map["level0"]}')
    # print(f'level1: {profile_map["level1"]}')
    # exit(0)
    return profile_map

def select_best_schedule_in_node_profile_data(file_name: str, local_size: int):
    # [NOTE]: Only for full attention
    # map_key: ((Sq, Skv), (Nhq, Nhg), bs, D, causal) -> Time[fwd/bwd]  # S per GPU !!!
    profile_map = {
        'level0': {},   # map_key -> [(Y,X,fused,Time), (Y,X,fused,Time)] # F/B
        'level1': {},
        # 'level2': {},
    }
    # YXs = [(1, 8), (2, 4), (4, 2), (8, 1)]
    YXs = []
    for y in range(1, local_size + 1):
        if local_size % y == 0:
            YXs.append((y, local_size // y))
    # print(f'YXs: {YXs}')
    
    baselines = ['ring_flash_attn_func', 'zigzag_ring_flash_attn_func', 'stripe_flash_attn_func']
    skip_lines = 0
    fused = 0
    YX_id = None
    pat0 = re.compile(r'^fob=(\d).*$')
    pat1 = re.compile(r'^.*SP=\((\d+),(\d+)\),.*S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat2 = re.compile(r'^.*iter/s, (-?(\d+(?:\.\d+)?(?:e[+-]\d+)?)) s/iter,.*$')
    pat3 = re.compile(r'^.*fused.*$')
    # non causal
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if skip_lines > 0:
                skip_lines -= 1
                continue
            for baseline in baselines:  # [NOTE]: Results of baseline models are error !!!
                if baseline in line:
                   skip_lines = 1
                #    print(f'baseline: {baseline}')
                   continue
            if 'orchestrated_attn_func' in line:
                cur_level = 0
                if 'fused' in line:
                    cur_level = 1
                # continue
            res = pat2.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                assert cur_map_key is not None and fob is not None
                if cur_map_key not in profile_map['level0']:
                    for l in range(2):
                        profile_map[f'level{l}'][cur_map_key] = [None] * 2
                # print(f'cur_map_key: {cur_map_key}, fob: {fob}, {float(res.group(1))}')
                cur_time = float(res.group(1))
                for l in range(cur_level, 2):
                    if profile_map[f'level{l}'][cur_map_key][fob] is None or cur_time < profile_map[f'level{l}'][cur_map_key][fob][- 1]:
                        profile_map[f'level{l}'][cur_map_key][fob] = YXs[YX_id] + (fused, cur_time)
                YX_id += 1
                continue
            res = pat3.match(line)
            if res:
                assert YX_id == None or YX_id == len(YXs)
                YX_id = 0
                fused = 1   # Convert to fused
                continue
            res = pat1.match(line)
            if res: # new map_key
                # print(f'res: {res.group(0)}, {res.group(1)}, {res.group(2)}, {res.group(3)}, {res.group(4)}, {res.group(5)}, {res.group(6)}, {res.group(7)}, {res.group(8)}, {res.group(9)}')
                SP = (int(res.group(1)), int(res.group(2)))
                tot_sp = SP[0] * SP[1]
                S = (int(res.group(3)) // tot_sp, int(res.group(4)) // tot_sp)
                # S = (int(res.group(3)), int(res.group(4)))  # just for analysis !!!
                Nh = (int(res.group(5)), int(res.group(6)))
                bs = int(res.group(7))
                D = int(res.group(8))
                causal = res.group(9) == 'True'
                cur_map_key = (SP, S, Nh, bs, D, causal)
                assert YX_id == None or YX_id == len(YXs)
                YX_id = 0
                fused = 0   # default: nonfused is ahead of fused
                # print(f'pat1, fob={fob}, cur_map_key: {cur_map_key}')
                continue
            res = pat0.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                fob = int(res.group(1))
                # print(f'fob: {fob}')
                continue
    profile_map['level2'] = profile_map['level1']   # [NOTE]: Passed for correction
    # causal
    # print(f'profile_map: {profile_map}')
    # if torch.distributed.get_rank() == 0:
    #     print(f'level0: {profile_map["level0"]}')
    #     print(f'level1: {profile_map["level1"]}')
    # exit(0)
    return profile_map

