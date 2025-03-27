#!/bin/bash

set -x

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

NET_DEVICE=$MLP_SOCKET_IFNAME
MLP_GPU=8
# NCCL_PATH=/common_libs/nccl_2.19.322/build/lib
MLP_MPI_HOSTFILE=/root/mpi_rack_hostfile
# NVSHMEM_PATH=/usr/local/nvshmem/lib
EXP_NAME='task2_BSA'

# export MLP_WORKER_NUM=4
# export MLP_WORKER_NUM=2
# export MLP_WORKER_NUM=1
# source $1

mkdir -p results
mkdir -p results/${EXP_NAME}

# Envs:
export CLUSTER_NAME=fit
export PLATFORM='A800'
export PLATFORM='H800'
# # set pulp tmp dir; [TODO]: `TMPDIR` is used by both pulp and openmpi
export PULP_TMPDIR=./search_algo/tmp
# export TMPDIR=search_algo/tmp
PYTHON_EXECUBLE=search_algo/task2_bsa.py
# torch.profiler:
export TRACE_NAME=${CLUSTER_NAME}
TB_DIR=./prof_results/tb
mkdir -p $TB_DIR
LOGGING_ARGS=""


mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --hostfile ${MLP_MPI_HOSTFILE} \
        --allow-run-as-root -oversubscribe -map-by ppr:8:node \
        --bind-to numa \
        -mca pml ob1 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=${NET_DEVICE} \
        --output-filename results/${TIMESTAMP} \
        -x NCCL_PXN_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_NET_GDR_LEVEL=4 \
        -x NCCL_IB_RETRY_CNT=7 \
        -x NCCL_IB_TIME_OUT=32 \
        -x NCCL_IB_QPS_PER_CONNECTION=8 \
        -x NCCL_P2P_LEVEL=NVL \
        -x NCCL_DEBUG=VERSION \
        -x PATH \
        -x MASTER_ADDR=$(cat $MLP_MPI_HOSTFILE | head -n 1 | sed -s 's/slots=8//g') \
        -x MASTER_PORT=${MLP_WORKER_0_PORT} \
        -x GLOO_SOCKET_IFNAME=${NET_DEVICE} \
        -x NCCL_SOCKET_IFNAME=${NET_DEVICE} \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        -x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -x NCCL_NVLS_ENABLE=0 \
        -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -x CLUSTER_NAME \
        -x PLATFORM \
        -x TRACE_NAME \
        ./scripts/runtime/bench_dist_attn.sh \
        python $PYTHON_EXECUBLE \
            $LOGGING_ARGS \
        2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log
        
        # bash ./scripts/wrapper.sh 2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log


set +x
exit 0
