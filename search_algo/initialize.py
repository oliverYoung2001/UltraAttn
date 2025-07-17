import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
import socket
import torch.distributed as dist
from search_algo.global_vars import *
from orchestrated_attn.global_vars import set_global_var as set_global_var_orch
from orchestrated_attn.global_vars import get_global_var as get_global_var_orch
import random
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from functools import partial
import math
from search_algo.database import Prof_DB
from datetime import timedelta
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def initialize_prof_db(gloo_global_group):
    # Generate Intra_Execution_Plans:
    CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
    prof_db = Prof_DB(CLUSTER_NAME, PLATFORM, gloo_global_group)
    return prof_db
    
def parse_slurm_tasks_per_node(tasks_per_node):
    # 4(x2), 8, ...
    return int(tasks_per_node.split('(')[0])

def get_proc_info():
    if os.getenv('SLURM_PROCID', None) is not None:    # launch with Slurm
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        ip = os.environ['SLURM_STEP_NODELIST']
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.environ['SLURM_CLUSTER_NAME']
        nodeid = int(os.environ['SLURM_NODEID'])
        nodename = os.environ['SLURMD_NODENAME']
        tasks_per_node = parse_slurm_tasks_per_node(os.environ['SLURM_TASKS_PER_NODE'])
        
    elif os.getenv('OMPI_COMM_WORLD_RANK', None) is not None: # launch with OpenMPI
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # ip = os.environ['SLURM_STEP_NODELIST']
        ip = None
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        # nodeid = int(os.environ['SLURM_NODEID'])
        # nodename = os.environ['SLURMD_NODENAME']
        nodename = None
        # tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
        tasks_per_node = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        nodeid = rank // tasks_per_node
        
    else:   # [NOTE]: assume that execute from python
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        hostname = socket.gethostname()
        nodename = None
        nodeid = 0
        world_size = 1
        tasks_per_node = 1
        rank = 0
        local_rank = 0
        hostip = socket.gethostbyname(hostname)
        ip = None
        # print(f'hostip: {hostip}')
        os.environ['MASTER_ADDR'] = hostip
        os.environ['MASTER_PORT'] = str(random.randint(0, 12000) + 10000)
        # os.environ['CLUSTER_NAME'] = 'fit'
        # os.environ['PLATFORM'] = 'A800'
        # raise Exception("Unknown Launcher !!!")
    proc_info = {
        'clustername': clustername,
        'hostname': hostname,
        'nodename': nodename,
        'nodeid': nodeid,
        'world_size': world_size,
        'tasks_per_node': tasks_per_node,
        'rank': rank,
        'local_rank': local_rank,
        'hostip': hostip,
        'ip': ip,
        'deviceid': local_rank,
    }
    proc_info['node_num'] = world_size // tasks_per_node
    # print(f'proc_info: {proc_info}')
    return proc_info

def initialize_distribution():
    PROC_INFO = get_proc_info()
    set_global_var(f'PROC_INFO', PROC_INFO)
    # print(f'PROC_INFO: {PROC_INFO}')
    
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    # MASTER_ADDR = 'localhost'
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
    global_ranks = list(range(PROC_INFO['world_size']))
    dist.init_process_group(
        backend=BACKEND,
        # init_method=init_method, 
        rank=PROC_INFO['rank'], 
        world_size=PROC_INFO['world_size'])
    gloo_global_group_timeout = timedelta(weeks=4)
    gloo_global_group = dist.new_group(ranks=global_ranks, backend='gloo', timeout=gloo_global_group_timeout)
    ncclcomm_global = PyNcclCommunicator(gloo_global_group, ranks=global_ranks, device=PROC_INFO['local_rank'])
    
    cpu_group_dict = get_global_var_orch('cpu_group_dict')
    cpu_group_dict[tuple(global_ranks)] = gloo_global_group
    set_global_var_orch('cpu_group_dict', cpu_group_dict)
    
    ncclcomm_dict = get_global_var_orch('ncclcomm_dict')
    ncclcomm_dict[tuple(global_ranks)] = ncclcomm_global
    set_global_var_orch('ncclcomm_dict', ncclcomm_dict)
    
    # Create all sub groups:
    # for nr in range(3, math.ceil(math.log2(PROC_INFO['world_size']))):        # [ERROR]
    # for nr in range(math.ceil(math.log2(PROC_INFO['world_size'])) - 1, 2, - 1): # OK
    for nr in range(math.ceil(math.log2(PROC_INFO['world_size'])) - 1, - 1, - 1): # OK
        sub_ranks = tuple(range(1 << nr))
        if torch.distributed.get_rank() in sub_ranks:
            gloo_global_group_sub = dist.new_group(ranks=sub_ranks, backend='gloo')
            ncclcomm_global_sub = PyNcclCommunicator(gloo_global_group_sub, ranks=sub_ranks, device=PROC_INFO['local_rank'])
            
            cpu_group_dict = get_global_var_orch('cpu_group_dict')
            cpu_group_dict[tuple(sub_ranks)] = gloo_global_group_sub
            set_global_var_orch('cpu_group_dict', cpu_group_dict)
            
            ncclcomm_dict = get_global_var_orch('ncclcomm_dict')
            ncclcomm_dict[tuple(sub_ranks)] = ncclcomm_global_sub
            set_global_var_orch('ncclcomm_dict', ncclcomm_dict)
        torch.distributed.barrier(gloo_global_group)
        

    # [NOTE]: we create a gloo global group because we use it to barrier in benchmark_orchestrate to prevent cudagraph overlapped with nccl ops !!!
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f'rank{rank}, world_size{world_size}, hostname: {socket.gethostname()}')
    # initialize_distributed()    # used by lightseq

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
        torch.cuda.set_device(device)
        # Create sync_tensor for placeholder_op
        SYNC_SIZE = 8 * pow(1024, 3) # 8GB
        # SYNC_SIZE = 4 * pow(1024, 3) # 8GB
        # sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=torch.cuda.current_device())
        # set_global_var_orch(f'sync_tensor', sync_tensor)
        set_global_var(f'SYNC_SIZE', SYNC_SIZE)
        # Initialize NVML
        nvmlInit()
    else:
        device = 'cpu'
    torch.distributed.barrier(gloo_global_group)
    return ncclcomm_global, gloo_global_group
