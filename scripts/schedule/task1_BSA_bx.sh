#!/bin/bash

set -x

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

# NET_DEVICE=$MLP_SOCKET_IFNAME
# MLP_GPU=8
GPUS_PER_NODE=8
# MLP_MPI_HOSTFILE=/root/mpi_rack_hostfile

# export MLP_WORKER_NUM=1
NNODES=1
HOST='g0002'
GPU_NUM=$((MLP_WORKER_NUM * $MLP_GPU))

# Envs:
# export CLUSTER_NAME=planck
# export CLUSTER_NAME=hamming
# export PLATFORM='H100'
export CLUSTER_NAME=bx
export PLATFORM='H800'
# # set pulp tmp dir; [TODO]: `TMPDIR` is used by both pulp and openmpi
export PULP_TMPDIR=./search_algo/tmp    # [DEPRECATED]
# export TMPDIR=search_algo/tmp
PYTHON_EXECUBLE=search_algo/task1_bsa.py

# Attention Patterns
MODEL_ARGS=()
if [ ! -z $1 ]; then
    MODEL_ARGS+=(
        --exp-class $1 # ['bsa_train', 'dense_train', 'bsa_infer']
    )
fi

# Logging ARGS:
export TRACE_NAME=${CLUSTER_NAME}
TB_DIR=./prof_results/tb
mkdir -p $TB_DIR
LOGGING_ARGS=""
# End

# Logs
EXP_NAME=task1_BSA_${CLUSTER_NAME}
mkdir -p results
mkdir -p results/${EXP_NAME}

# Envs for gurobipy
# if [ $CLUSTER_NAME == 'planck' ]; then
#     export GUROBI_NUM_THREADS=64
# elif [ $CLUSTER_NAME == 'hamming' ]; then
#     export GUROBI_NUM_THREADS=96
# fi
GUROBI_NUM_THREADS=128

# Envs For Nsight
NSYS_DIR=./prof_results/nsys_orchestrate
mkdir -p $NSYS_DIR
NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPU_NUM}_$(date "+%Y%m%d-%H%M%S")"
NSIGHT_CMD=""

time \
$NSIGHT_CMD \
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
    -x GUROBI_NUM_THREADS \
    -x TRACE_NAME \
    ./scripts/runtime/bench_dist_attn.sh \
    python $PYTHON_EXECUBLE \
        $LOGGING_ARGS \
        ${MODEL_ARGS[@]} \
    2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log

set +x
exit 0
