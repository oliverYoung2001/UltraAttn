import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from search_algo.search_engine import Dist_Attn_Config, Evaluation_Configs
from search_algo.execute_plan import Execution_Plan, Fused_Execution_Plan
from search_algo.dependent_graph import Cuda_Kernel, Comp_Kernel, Comm_Kernel
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from typing import Optional, Union
from orchestrated_attn.utils import *
from search_algo.utils import filter_kwargs, calc_flops, all_wait_main_stream, main_stream_wait_all, report_memory
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_backward
from search_algo.global_vars import get_global_var as get_global_var_search_algo
from search_algo.database import Prof_DB
from search_algo.bsa_utils import convert_intra_profile_key_to_exe_key_or_plan, bsa_is_causal
import json
import time
import pickle

def prepare_inter_comp_plans(inter_bsa_execution_plan: Execution_Plan, prof_db: Prof_DB, da_config: Dist_Attn_Config) -> dict:   # [TODO]
    # OBJ1: build inter_comp_plans_dict
    # OBJ2: Set correct execution_plan to each inter kernel
    
    # inter_comp_plans_dict: {inter_comp_key: intra_bsa_exe_plan}
    # inter_comp_key = ((relative_Sq, relative_Skv), str(da_config.bsa_config.bsa_repr)) # [Deprecated], used only in inter_CP=1 !!!
    # inter_comp_key = ((relative_Sq, relative_Skv), key in 'intra_bsa_exe_plans_profile.json' or 'inter_bsa_exe_plans_kv.json' 
    #                       or inter_CP=1 of 'inter_bsa_exe_plans_kv.json')
    PROC_INFO = get_global_var_search_algo(f'PROC_INFO')
    node_id = PROC_INFO['nodeid']
    with open(prof_db.INTRA_BSA_EXE_PLANS_KV, 'r') as f:
        intra_bsa_exe_plans_dict = json.load(f)
    inter_comp_plans_dict = {}
    
    def get_exe_plan_from_exe_key_or_plan(exe_key_or_plan: Union[str, Fused_Execution_Plan]):
        nonlocal intra_bsa_exe_plans_dict, inter_comp_plans_dict
        if isinstance(exe_key_or_plan, str):
            assert exe_key_or_plan in intra_bsa_exe_plans_dict.keys(), f'[ERROR]: Exe_plan of inter_comp_key={exe_key_or_plan} is not cached !!!'
            plan_id = intra_bsa_exe_plans_dict[exe_key_or_plan]
            if inter_comp_key not in inter_comp_plans_dict.keys():
                with open(f'{prof_db.INTRA_BSA_EXE_PLANS_DIR}/{plan_id}.pkl', 'rb') as fin:
                    intra_bsa_execution_plan: Execution_Plan = pickle.load(fin)
            else:
                intra_bsa_execution_plan = inter_comp_plans_dict[inter_comp_key]
        else:   # Fused_Execution_Plan
            assert isinstance(exe_key_or_plan, Fused_Execution_Plan)
            intra_bsa_execution_plan = exe_key_or_plan
        return intra_bsa_execution_plan
    
    if isinstance(inter_bsa_execution_plan, Fused_Execution_Plan) or \
        (inter_bsa_execution_plan.d_graph is not None and inter_bsa_execution_plan.d_graph.seqlen_variable_graph):
        inter_bsa_execution_plan.materialize(da_config)
    # Now 'inter_bsa_execution_plan' is materialized !!!
    assert inter_bsa_execution_plan.is_materialized() == True
    
    # [NOTE]: Ensure that kernels with the same inter_comp_key using the same exe_plan object
    if isinstance(inter_bsa_execution_plan, Fused_Execution_Plan):  # Only one inter_comp_plan per node. For full w kernel tile
        batch_degrees = (inter_bsa_execution_plan.X, inter_bsa_execution_plan.Y)
        kernel_list = inter_bsa_execution_plan.gpu_kernel_lists[node_id]
        assert len(kernel_list) == 1, f'Fused_Execution_Plan only has one comp kernel'
        profile_comp_map_key = kernel_list[0].comp_map_key # [TODO]: deal with seqlen_variable_graph, select from all ablations
        inter_comp_key = (batch_degrees, profile_comp_map_key)
        exe_key_or_plan = convert_intra_profile_key_to_exe_key_or_plan(profile_comp_map_key, prof_db)
        intra_bsa_execution_plan = get_exe_plan_from_exe_key_or_plan(exe_key_or_plan)
        # Record intra_bsa_execution_plan
        inter_bsa_execution_plan.kernel_execution_plan = intra_bsa_execution_plan
        inter_comp_plans_dict[inter_comp_key] = intra_bsa_execution_plan
    else:
        for kernel in inter_bsa_execution_plan.gpu_kernel_lists[node_id]:
            if not isinstance(kernel, Comp_Kernel):
                continue
            profile_comp_map_key = kernel.comp_map_key  
            #   Create inter_comp_key which is **identifier** for clear/real intra workload
            rcs = kernel.key[2: 4]
            batch_degrees = (len(rcs[0]) if isinstance(rcs[0], tuple) else 1, len(rcs[1]) if isinstance(rcs[1], tuple) else 1)
            inter_comp_key = (batch_degrees, profile_comp_map_key)  # [TODO]: parse bsa_repr ???
            #   End
            exe_key_or_plan = convert_intra_profile_key_to_exe_key_or_plan(profile_comp_map_key, prof_db)
            intra_bsa_execution_plan = get_exe_plan_from_exe_key_or_plan(exe_key_or_plan)
            # Record intra_bsa_execution_plan
            kernel.execution_plan = intra_bsa_execution_plan
            inter_comp_plans_dict[inter_comp_key] = intra_bsa_execution_plan
    return inter_comp_plans_dict

def benchmark_ops(streams, global_group, device, f, inputs, \
                warmup, warmup_cudagraph, num_iter, use_cudagraph, TRACE_NAME, args, placeholder_op):
    # print_rank_0(f'streams: {streams}')
    torch.cuda.empty_cache()
    main_stream = streams['intra'][0] # intra comp stream
    stream_list = streams['intra'] + streams['inter']
    # warmup
    placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead

    # preprocess
    all_wait_main_stream(stream_list, main_stream)
    with torch.no_grad():
        for _ in range(warmup):
            _ = f(**inputs)
    # postprocess
    main_stream_wait_all(stream_list, main_stream)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)   
    # # [NOTE]: we don't barrier here to prevent WARN of 
    # # "[W CUDAGraph.cpp:145] Warning: Waiting for pending NCCL work to finish before starting graph capture. (function operator())"
    # print_rank_0(f'Warmup done !!!')

    assert use_cudagraph == False, "Not support cudagraph in this version !!!"
    if use_cudagraph:
        if args.profiler_with_tensorboard:
            with torch.profiler.profile():  # workaround of issue 75504 of PyTorch
                pass
        # Capture cuda graph
        # torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        # return
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=streams[0]):
            pass
            # preprocess
            for stream in streams[1:]:
                stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                _ = f(**inputs)
            # postprocess
            for stream in streams[1:]:
                torch.cuda.current_stream().wait_stream(stream)
        
    is_runned = False
    
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    t1 = time.time()

    # warmup cudagraph
    if use_cudagraph:
        for _ in range(warmup_cudagraph):
            g.replay()
            # torch.cuda.synchronize()
            # torch.distributed.barrier(group=global_group)
        # print_rank_0(f'Warmup cudagraph done !!!')

                
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    
    t2 = time.time()
    td = - 1
    # assert args.profiler_with_tensorboard == False, "Not support profiler_with_tensorboard in this version !!!"
    # if args.profiler_with_tensorboard and not hasattr(args, "tb_profiled"):
    if args.profiler_with_tensorboard:
        args.tb_profiled = True
        is_runned = True
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=f'{args.tb_dir}', 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter in range(TOTAL_TURNS):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                if use_cudagraph:
                    g.replay()
                else:
                    if iter % BARRIER_FREQ == 0:
                        placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
                    # preprocess
                    all_wait_main_stream(stream_list, main_stream)
                    with torch.no_grad():
                        _ = f(**inputs)
                    # postprocess
                    main_stream_wait_all(stream_list, main_stream)
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier(group=global_group)
                prof.step()
        
        num_iter = TOTAL_TURNS
        
    if not is_runned: 
        # run
        if use_cudagraph:
            for _ in range(num_iter):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                g.replay()
                # torch.cuda.synchronize()    # almost no effect on performance
                # torch.distributed.barrier(group=global_group)   # 64TFlops -> 43TFlops
        else:
            for i in range(3):
                # print_rank_0(f'step{i}:')
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
                event_start.record(stream=main_stream)
                # # preprocess
                # all_wait_main_stream(stream_list, main_stream)
                with torch.no_grad():
                    for _ in range(num_iter):
                        # preprocess
                        all_wait_main_stream(stream_list, main_stream)
                        _ = f(**inputs)
                        # postprocess
                        main_stream_wait_all(stream_list, main_stream)
                # # postprocess
                # main_stream_wait_all(stream_list, main_stream)
                event_end.record(stream=main_stream)
                torch.cuda.synchronize()
                td = event_start.elapsed_time(event_end) / 1000 # s

    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    t3 = time.time()
    if td < 0:
        td = t3 - t2
    else:
        td = torch.tensor(td)   # On cpu
        torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MAX, group=global_group, async_op=False)
        torch.cuda.synchronize()
        td = td.cpu().item()
    torch.cuda.empty_cache()
    return t1, t2, t3, td

def create_buf_dict(da_config: Dist_Attn_Config, exp_config: Evaluation_Configs, \
                    execution_plan: Union[Execution_Plan, Fused_Execution_Plan], fused: bool, \
                    batch_degrees: tuple, tensor_buf: torch.Tensor, level_rank: int) -> dict:
    # buf_dict stores absolute shape infos of kernels of the exe_plan
    assert len(batch_degrees) == 2, f'Invalid batch_degrees: {batch_degrees}'   # [Q_batch_degree, KV_batch_degree]
    bs = da_config.bs
    Sq, Skv = da_config.S_per_partition
    Sq *= batch_degrees[0]
    Skv *= batch_degrees[1]
    Nhq, Nhg = da_config.Nh
    d = da_config.D
    # print_rank_0(f'da_config.S_per_partition: {da_config.S_per_partition}, (Sq, Skv): ({Sq}, {Skv})')
    
    fob = exp_config.fob
    # buf_dict : (('i'/'o', 'r'/'c'), batch_degree) -> Integrated_Data
    if not fused:
        integrated_data_types = {
            ('i', 'r'): Input_Row_Fwd if fob == 0 else Input_Row_Bwd,
            ('i', 'c'): Input_Col_Fwd if fob == 0 else Input_Col_Bwd,
            ('o', 'r'): Output_Row_Fwd if fob == 0 else Output_Row_Bwd,
            ('o', 'c'): Output_Col_Fwd if fob == 0 else Output_Col_Bwd,
        }
        # Q_buf = tensor_buf[2 * bs * Skv * Nhg * d * 2: ]    # last 2 stands for the largest low_dg of all execution plans !!!
        Q_buf = tensor_buf
        KV_buf = tensor_buf
        # batch_degrees = (1, 1):   # for input/output and comm kernels
        buf_dict = {
            (('i', 'r'), 1): integrated_data_types[('i', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * 1),
            (('i', 'c'), 1): integrated_data_types[('i', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * 1),
            (('o', 'r'), 1): integrated_data_types[('o', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * 1),
            (('o', 'c'), 1): integrated_data_types[('o', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * 1),
        }
        for kernel in execution_plan.gpu_kernel_lists[level_rank]:
            # [NOTE]: Ignore Comm Kenrel here, because kernel fusion of comm kernel is not supported now
            if isinstance(kernel, Comp_Kernel):
                rcs = kernel.key[2: 4]
                low_bds = (
                    len(rcs[0]) if isinstance(rcs[0], tuple) else 1,
                    len(rcs[1]) if isinstance(rcs[1], tuple) else 1,
                )
                if (('i', 'r'), low_bds[0]) not in buf_dict.keys():
                    buf_dict[(('i', 'r'), low_bds[0])] = integrated_data_types[('i', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * low_bds[0])
                    buf_dict[(('o', 'r'), low_bds[0])] = integrated_data_types[('o', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * low_bds[0])
                if (('i', 'c'), low_bds[1]) not in buf_dict.keys():
                    buf_dict[(('i', 'c'), low_bds[1])] = integrated_data_types[('i', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * low_bds[1])
                    buf_dict[(('o', 'c'), low_bds[1])] = integrated_data_types[('o', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * low_bds[1])
        
        buf_dict['graph_type'] = 'general'
        buf_dict['tensor_buf'] = tensor_buf
        buf_dict['inp_row'] = buf_dict[(('i', 'r'), 1)]
        buf_dict['inp_col'] = buf_dict[(('i', 'c'), 1)]
        buf_dict['out_row'] = buf_dict[(('o', 'r'), 1)]
        buf_dict['out_col'] = buf_dict[(('o', 'c'), 1)]
    else:
        assert isinstance(execution_plan, Fused_Execution_Plan), f'Invalid execution_plan: {execution_plan}'
        Y, X = execution_plan.Y, execution_plan.X
        if fob == 0:    # forward
            ir_nelems = bs * Sq * Nhq * d   # q
            ic_nelems = bs * Skv * Nhg * (d * 2)   # k, v
            or_nelems = bs * Sq * Nhq * d   # o, (lse)
            oc_nelems = 0
        else:   # backward
            ir_nelems = bs * Sq * Nhq * (d * 2 + 1 * (2 + 1))   # q, do, D, lse
            ic_nelems = bs * Skv * Nhg * (d * 2)   # k, v
            or_nelems = bs * Sq * Nhq * d        # dq
            oc_nelems = bs * Skv * Nhg * (d * 2)  # dk, dv

        ir_tot = ir_nelems * X
        ic_tot = ic_nelems * Y
        or_tot = or_nelems * X
        oc_tot = oc_nelems * Y
        # print_rank_0(f'ir_tot: {ir_tot}, ic_tot: {ic_tot}, or_tot: {or_tot}, oc_tot: {oc_tot}')
        buf = tensor_buf[: ir_tot + ic_tot + or_tot + oc_tot]
        cur_offset = ir_tot + ic_tot + or_tot + oc_tot
        # buf = torch.empty(ir_tot + ic_tot + or_tot + oc_tot, dtype=DTYPE, device=torch.cuda.current_device())
        # print_rank_0(f'buf: {buf.numel() * 2} B')
        buf_dict = {
            'ir': buf[: ir_tot],
            'ic': buf[ir_tot: ir_tot + ic_tot],
            'or': buf[ir_tot + ic_tot: ir_tot + ic_tot + or_tot],
            'oc': buf[ir_tot + ic_tot + or_tot: ],
            'ir_nelems': ir_nelems,
            'ic_nelems': ic_nelems,
            'or_nelems': or_nelems,
            'oc_nelems': oc_nelems,
            'ir_tot': ir_tot,
            'ic_tot': ic_tot,
            'or_tot': or_tot,
            'oc_tot': oc_tot,
            'graph_type': 'fused',
        }
        if fob == 1:
            # inp_row_extra_buf = torch.empty(ir_tot, dtype=DTYPE, device=torch.cuda.current_device())
            inp_row_extra_buf = tensor_buf[cur_offset: cur_offset + ir_tot]
            cur_offset += ir_tot
            buf_dict['ir_'] = buf_dict['ir']
            buf_dict['ir'] = inp_row_extra_buf
        ir_class = Input_Row_Fwd if fob == 0 else Input_Row_Bwd
        ic_class = Input_Col_Fwd if fob == 0 else Input_Col_Bwd
        cur_x_id = level_rank % X   # [0, X)
        cur_y_id = level_rank // X  # [0, Y)
        buf_dict['inp_row'] = ir_class.from_da_config_with_buf(da_config=da_config, buf=buf_dict['ir'][ir_nelems * cur_x_id: ir_nelems * (cur_x_id + 1)], batch_degree=batch_degrees[0])
        buf_dict['inp_col'] = ic_class.from_da_config_with_buf(da_config=da_config, buf=buf_dict['ic'][ic_nelems * cur_y_id: ic_nelems * (cur_y_id + 1)], batch_degree=batch_degrees[1])
    return buf_dict

def benchmark_orchestrate_bsa(args, raw_f, da_config: Dist_Attn_Config, exp_config: Evaluation_Configs, tensor_buf: torch.Tensor,
                              inter_bsa_execution_plans: Optional[List[Execution_Plan]], inter_comp_plans_dicts: Optional[List[dict]],
                              warmup=11, num_iter=20, log=True, global_group=None, ncclcomm_global: PyNcclCommunicator = None, 
                              use_cudagraph=False, mode='test'):
    # [TODO]: Use only `fob` in exp_configs for intra profiling !!!
    # [TODO]: Remove the usage of mode !!!
    assert len(inter_bsa_execution_plans) == len(inter_comp_plans_dicts), 'inter_bsa_execution_plans and inter_comp_plans_dicts must be bijective !!!'
    # print_rank_0(f'[INFO]: use_cudagraph: {use_cudagraph}')
    warmup_cudagraph = 100    # [NOTE]: cudagraph is deprecated
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    PROC_INFO = get_global_var_search_algo(f'PROC_INFO')
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    # print(f'rank{rank}, node_id: {node_id}, node_num: {node_num}', flush=True)
    if rank == 0:
        print(f'# {raw_f.__name__}', flush=True)
    world_size = torch.distributed.get_world_size(group=global_group)
    # device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    # torch.cuda.set_device(device)

    # Configs:
    batch_size = da_config.bs
    # Sq, Skv = da_config.S_per_gpu   # Sq, Skv per gpu !!!
    nheads = da_config.Nh[0]    # Nh, Ng
    d = da_config.D
    dropout_p = 0
    deterministic = False

    assert d % 8 == 0
    
    def create_inputs_and_buf_dict(exp_config: Evaluation_Configs, inter_execution_plan: Execution_Plan, inter_comp_plans: dict):
        # qkvdo = tensor_buf[: 4 * batch_size * seqlen * nheads * d].view(4 * batch_size, seqlen, nheads, d)
        # # qkv.requires_grad = True
        # q, k, v, do = qkvdo.chunk(4, dim=0)
        # D_buf = tensor_buf[4 * batch_size * seqlen * nheads * d: ]
        # D = D_buf[: 2 * batch_size * seqlen * nheads * 1].view(FULL_DTYPE).view(batch_size, nheads, seqlen)   # [mbs, Nh, S], torch.float32, 2 stands for 2 torch.bfloat16
        # lse_buf = D_buf[2 * batch_size * seqlen * nheads * 1: ]
        # # q, k, v, do, o = qkvdoo.chunk(5, dim=0)
        # # lse_buf = tensor_buf[5 * batch_size * seqlen * nheads * d: ]
        # lse = lse_buf[: batch_size * seqlen * nheads * 1].view(batch_size, nheads, seqlen)   # [mbs, Nh, S]
        
        # Create buf_dict for inter_comp_plans
        # exp_config: fob✅
        extra_dict = {
            'da_config': da_config,
            'exp_configs': exp_config,
        }
        inter_buf_dict = create_buf_dict(
            da_config, exp_config, inter_execution_plan, 
            isinstance(inter_execution_plan, Fused_Execution_Plan), 
            (1, 1), tensor_buf, node_id)
        inter_execution_plan.buf_dict = inter_buf_dict
        # Create buf_dict for inter_comp_plans
        for inter_comp_key, intra_execution_plan in inter_comp_plans.items():
            intra_execution_plan.buf_dict = create_buf_dict(    # [TODO]
                da_config, exp_config, intra_execution_plan, 
                isinstance(intra_execution_plan, Fused_Execution_Plan), 
                inter_comp_key[0], tensor_buf, local_rank)
        
        inputs = {
            "inp_row": inter_buf_dict['inp_row'], 
            "inp_col": inter_buf_dict['inp_col'],
            "dropout_p": dropout_p,
            "causal": None,
            "deterministic": deterministic,
            "sm_scale": d ** (-0.5),
            "softmax_scale": d ** (-0.5),
            "PROC_INFO": PROC_INFO,
            # 'buf_dict': buf_dict,
            'extra_dict': extra_dict,
        }
        return inputs

    bench_results = []
    if True:
        # exp_config: fob✅, (plan_path, inter_comp_profile_map) are useless for profiling
        fob = exp_config.fob
        
        torch.cuda.empty_cache()
        t0 = time.time()
        SP = da_config.SP   # (inter, intra)
        Ss = da_config.S    # S total
        Nhs = (nheads, nheads)
        bs = batch_size
        
        if fob == 0:
            f = raw_f
        else:
            f = orchestrated_attn_backward
        
        for inter_execution_plan, inter_comp_plans in zip(inter_bsa_execution_plans, inter_comp_plans_dicts):
            inputs = create_inputs_and_buf_dict(exp_config, inter_execution_plan, inter_comp_plans)
            # inputs['causal'] = causal
            # Mark in_ranks on execution_plans to judge whether kernel is on current rank easily
            #   for inter-level `kernels`
            if isinstance(inter_execution_plan, Execution_Plan):
                for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
                    kernel.in_ranks = set([rank])
            #   for intra-level kernels
            for _, intra_execution_plan in inter_comp_plans.items():
                if not isinstance(intra_execution_plan, Execution_Plan):
                    continue
                for kernel in intra_execution_plan.gpu_kernel_lists[local_rank]:
                    kernel.in_ranks = set([rank])
            # Modify da_config
            if isinstance(inter_execution_plan, Execution_Plan) and inter_execution_plan.da_config is not None:
                # [TODO]: hidden danger here !!! I don't know why we need da_config her
                assert inter_execution_plan.da_config.S == Ss
                assert inter_execution_plan.da_config.Nh == Nhs
                if inter_execution_plan.da_config.S != Ss:
                    inter_execution_plan.da_config.S = Ss
                if inter_execution_plan.da_config.Nh != Nhs:
                    inter_execution_plan.da_config.Nh = Nhs
            # Create streams for both inter and intra execution plans
            if not is_exist_global_var('streams'):
                streams = {
                    'inter': [],    # Comms
                    'intra': [],    # Comp, Comms
                }
            else:
                streams = get_global_var('streams')
            #   Create Streams for Inter Execution Plan
            if len(streams['inter']) < inter_execution_plan.stream_num:
                priorities = [0, - 1, - 2]
                priorities = [0] * inter_execution_plan.stream_num
                for _ in range(len(streams['inter']), inter_execution_plan.stream_num):
                    streams['inter'].append(torch.cuda.Stream(torch.cuda.current_device(), priority=priorities[_]))
            #   Create Streams for Intra Execution Plans
            for _, intra_execution_plan in inter_comp_plans.items():
                if not isinstance(intra_execution_plan, Execution_Plan):
                    intra_stream_num = 3
                else:
                    intra_stream_num = intra_execution_plan.stream_num
                if len(streams['intra']) < intra_stream_num:
                    priorities = [0, - 1, - 2]
                    priorities = [0] * intra_stream_num
                    for _ in range(len(streams['intra']), intra_stream_num):
                        streams['intra'].append(torch.cuda.Stream(torch.cuda.current_device(), priority=priorities[_]))
            set_global_var('streams', streams)
                
            # Set streams for each kernel of both inter and intra execution plans
                # Set Streams for Intra Execution Plans
            for _, intra_execution_plan in inter_comp_plans.items():
                if not isinstance(intra_execution_plan, Execution_Plan):
                    continue
                for kernel in intra_execution_plan.gpu_kernel_lists[local_rank]:
                    if isinstance(kernel, Comp_Kernel):
                        kernel.stream = streams['intra'][0]
                    else:
                        if kernel.key[3] == local_rank:
                            kernel.stream = streams['intra'][1]    # Send
                        else:
                            kernel.stream = streams['intra'][2]    # Recv
                # Set Streams for Inter Execution Plan
            if isinstance(inter_execution_plan, Execution_Plan):
                for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
                    # print_rank_0(f'inter kernel: {kernel.key}, {isinstance(kernel, Comp_Kernel)}, {isinstance(kernel, Comm_Kernel)}')
                    if isinstance(kernel, Comp_Kernel):
                        # kernel.stream = streams['inter'][0]
                        assert len(streams['intra']) >= 3
                        kernel.sub_streams = streams['intra'][: 3] # [NOTE]: hardcode here !!! not support ablation0 !!!
                        assert not hasattr(kernel, 'stream')
                    else:
                        if kernel.key[3] == node_id:
                            kernel.stream = streams['inter'][1]    # Send
                        else:
                            kernel.stream = streams['inter'][2]    # Recv
                        assert not hasattr(kernel, 'sub_streams')
            torch.cuda.synchronize()
            torch.distributed.barrier(group=global_group)
            
            # Build nccl communicator for each pair of ranks
            ncclcomm_dict = get_global_var('ncclcomm_dict')
                # For Inter
            if isinstance(inter_execution_plan, Fused_Execution_Plan):
                # [TODO]
                # Create Row&Col PyNcclCommunicator
                Y = inter_execution_plan.Y
                X = inter_execution_plan.X
                cur_x_id = node_id % X
                cur_y_id = node_id // X
                # r_key = tuple(range(node_id * local_size + cur_y_id * X, node_id * local_size + (cur_y_id + 1) * X))
                # c_key = tuple(range(node_id * local_size + cur_x_id, (node_id + 1) * local_size, X))
                r_key = tuple(range(cur_y_id * X * local_size + local_rank, (cur_y_id + 1) * X * local_size + local_rank, local_size))
                c_key = tuple(range(cur_x_id * local_size + local_rank, world_size, X * local_size))
                # print(f'rank{rank}, r_key: {r_key}, c_key: {c_key}', flush=True)
                assert rank in r_key and rank in c_key
                if r_key not in ncclcomm_dict.keys():
                    ncclcomm_dict[r_key] = PyNcclCommunicator(global_group, ranks=r_key, device=torch.cuda.current_device())
                if c_key not in ncclcomm_dict.keys():
                    ncclcomm_dict[c_key] = PyNcclCommunicator(global_group, ranks=c_key, device=torch.cuda.current_device())
            else:
                for kernel in inter_execution_plan.valid_kernels:
                    if isinstance(kernel, Comm_Kernel):
                        key = (kernel.key[3] * local_size + local_rank, kernel.key[4] * local_size + local_rank)    # (send, recv)
                        if rank in key:
                            if key not in ncclcomm_dict.keys():
                                ncclcomm_dict[key] = PyNcclCommunicator(global_group, ranks=key, device=torch.cuda.current_device())
                            kernel.ncclcomm = ncclcomm_dict[key]
                        torch.cuda.synchronize()
                        torch.distributed.barrier(group=global_group)
                # For Intra
            for _, intra_execution_plan in inter_comp_plans.items():
                    # print(f'rank{rank}, batch_degrees: {batch_degrees}, intra_execution_plan: {intra_execution_plan}', flush=True)
                if not isinstance(intra_execution_plan, Execution_Plan):    # fused intra execution plan
                    assert isinstance(intra_execution_plan, Fused_Execution_Plan)
                    # Create Row&Col PyNcclCommunicator
                    Y = intra_execution_plan.Y
                    X = intra_execution_plan.X
                    # print(f'rank{rank}, batch_degrees: {batch_degrees}, intra_execution_plan: {intra_execution_plan}', flush=True)
                    cur_x_id = local_rank % X
                    cur_y_id = local_rank // X
                    r_key = tuple(range(node_id * local_size + cur_y_id * X, node_id * local_size + (cur_y_id + 1) * X))
                    c_key = tuple(range(node_id * local_size + cur_x_id, (node_id + 1) * local_size, X))
                    assert rank in r_key and rank in c_key
                    if r_key not in ncclcomm_dict.keys():
                        ncclcomm_dict[r_key] = PyNcclCommunicator(global_group, ranks=r_key, device=torch.cuda.current_device())
                    if c_key not in ncclcomm_dict.keys():
                        ncclcomm_dict[c_key] = PyNcclCommunicator(global_group, ranks=c_key, device=torch.cuda.current_device())
                else:                                                      # non-fused intra execution plan
                    for kernel in intra_execution_plan.valid_kernels:
                        if isinstance(kernel, Comm_Kernel):
                            key = (node_id * local_size + kernel.key[3], node_id * local_size + kernel.key[4])    # (send, recv)
                            # report_memory(f'In nccl comm')
                            if rank in key:
                                if key not in ncclcomm_dict.keys():
                                    ncclcomm_dict[key] = PyNcclCommunicator(global_group, ranks=key, device=torch.cuda.current_device())
                                kernel.ncclcomm = ncclcomm_dict[key]
                            # torch.cuda.synchronize()
                            # torch.distributed.barrier(group=global_group)
            set_global_var('ncclcomm_dict', ncclcomm_dict)
            # set_global_var('cpu_group_dict', cpu_group_dict)
            # print kernel orders
            # for r in range(local_size):
            #     print_rank_0(f'rank{r}:')
            #     for kernel in execution_plan.gpu_kernel_lists[r]:
            #         # if isinstance(kernel, Comp_Kernel) or kernel.key[- 2] == 'o': # comm + output comm
            #         if kernel.key[- 2] == 'i':  # only input comm
            #             print_rank_0(f'{kernel.key}')
            
            inputs['execution_plan_dict'] = {
                'inter': inter_execution_plan,
                'intra': inter_comp_plans,
            }
            inputs = filter_kwargs(f, inputs)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=global_group)
            # continue
            TRACE_NAME = f'{os.environ["TRACE_NAME"]}_SP({node_num},{local_size})_w{world_size}_r{rank}_' \
                         f'S({da_config.S_per_partition[0]},{da_config.S_per_partition[1]})_bs{batch_size}_Nh{nheads}_D{nheads}_' \
                         f'{"causal" if True else "noncausal"}_{f.__name__}'    # [TODO]: Correct it !!!
                        #  f'{"causal" if causal else "noncausal"}_{f.__name__}'
            # Build placeholder_op
            SYNC_SIZE = get_global_var_search_algo('SYNC_SIZE')
            placeholder_op = partial(ncclcomm_global.all_reduce, tensor_buf[: SYNC_SIZE // tensor_buf.element_size()])
            # End
            # report_memory(f'Before benchmark_ops')
            t1, t2, t3, td = benchmark_ops(streams, global_group, torch.cuda.current_device(), f, inputs, \
                warmup, warmup_cudagraph, num_iter, use_cudagraph, TRACE_NAME, args, placeholder_op)
            # report_memory(f'After benchmark_ops')
            if rank == 0:
            # if True:
                assert da_config.bsa_config is not None, f'Now only bsa_config is supported'
                # if da_config.bsa_config:
                total_sparsity = da_config.bsa_config.total_sparsity
                # else:
                #     total_sparsity = 0.5 if causal else 1
                m_flops, h_flops = calc_flops(batch_size, da_config.S, nheads, d, fob=fob, total_sparsity=total_sparsity)
                mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
                # print(f"suffix: {plan_path.split('/')[-1]}, mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)
                if log:
                    print(f"mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, "
                        f"{td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)
                bench_results.append({
                    'hfu': hfu, 
                    'time': f'{td / num_iter:.3e}'
                })    # TFlops/s, s
            else:
                bench_results.append(None) 
    return bench_results
