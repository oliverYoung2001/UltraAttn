#!/bin/bash

set -x

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

NET_DEVICE=$MLP_SOCKET_IFNAME
MLP_GPU=8
# NCCL_PATH=/common_libs/nccl_2.19.322/build/lib
MLP_MPI_HOSTFILE=/root/mpi_rack_hostfile
# NVSHMEM_PATH=/usr/local/nvshmem/lib
EXP_NAME='task1_BSA'

export MLP_WORKER_NUM=1
# source $1

mkdir -p results
mkdir -p results/${EXP_NAME}

# Envs:
export CLUSTER_NAME=zhipu_planck
export PLATFORM='H100'
# # set pulp tmp dir; [TODO]: `TMPDIR` is used by both pulp and openmpi
export PULP_TMPDIR=./search_algo/tmp
# export TMPDIR=search_algo/tmp
PYTHON_EXECUBLE=search_algo/task1_bsa.py
# torch.profiler:
export TRACE_NAME=${CLUSTER_NAME}
TB_DIR=./prof_results/tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

MPI_EXTRA=''
if [ $CLUSTER_NAME == 'zhipu_planck' ]; then    # fix for planck
   MPI_EXTRA="$MPI_EXTRA \
   -mca oob_tcp_if_include 10.102.2.0/24 \
   "
fi

# Envs for gurobipy
if [ $CLUSTER_NAME == 'zhipu_planck' ]; then
    export GUROBI_NUM_THREADS=64
fi

mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --hostfile ${MLP_MPI_HOSTFILE} \
        --allow-run-as-root -oversubscribe -map-by ppr:8:node \
        --bind-to numa \
        -mca pml ob1 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=${NET_DEVICE} \
        $MPI_EXTRA \
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
        ./scripts/runtime/bench_dist_attn.sh \
        python $PYTHON_EXECUBLE \
            $LOGGING_ARGS \
        2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log
        
        # bash ./scripts/wrapper.sh 2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log


set +x
exit 0


# ./scripts/schedule/cpu_task_${CLUSTER_NAME}.sh \
if [[ $(hostname) == *nico* ]]; then
    EXE_MODE="CPUS"
else
    EXE_MODE="GPUS"
fi
# echo "EXE_MODE: ${EXE_MODE}"

export CLUSTER_NAME=fit
export PLATFORM='A800'
GPUS_PER_NODE=1
GPUS_PER_NODE=8
CPU_PER_TASK=13
PARTITION=a01
NNODES=1
HOST="g07"
HOST="g10"

# PARTITION=h01
# NNODES=1
# HOST="g40"

# HOST=None

export MASTER_PORT=$((RANDOM % 12000 + 10000))

MEM_PER_CPU=256G
# --mem-per-cpu $MEM_PER_CPU \
MEM_PER_NODE=256G

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$GPUS_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

# torch.profiler:
export TRACE_NAME=${CLUSTER_NAME}
TB_DIR=./prof_results/tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

# LOGGING_ARGS="${LOGGING_ARGS} \
# --profiler-with-tensorboard \
# --tb-dir $TB_DIR \
# "

# Nsight System:
export USE_NSYS=True
export USE_NSYS=False
export NSYS_DIR=./prof_results/nsys_orchestrate
mkdir -p $NSYS_DIR
# NSIGHT_CMD=""
# NSIGHT_CMD="nsys profile --trace=cuda,nvtx,osrt,mpi,nvtx --output=${TB_DIR}/nsys_${TRACE_NAME}.qdrep"

# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET




# Run with MPI
# salloc -N 1 -n 128 --gres=gpu:8 --cpus-per-task=??? --exclusive -p rag
# salloc -N 1 --gres=gpu:8 --cpus-per-task=104 --exclusive -p a01 -w g10

# Qiyuan
GPU_NUM=16
HOST_CONFIG="g3021:8,g3022:8"
GPU_NUM=8
HOST_CONFIG="g3017:8"
GPU_NUM=4
HOST_CONFIG="g3017:4"
HOST_CONFIG="g4008:4"
# HOST_CONFIG="g3029:4"
# HOST_CONFIG="g3027:2,g4003:2"
# HOST_CONFIG="g3021:2,g3022:2"

# Fit
if [[ $EXE_MODE == 'CPUS' ]]; then
    NUM_TASKS=8
    HOST_CONFIG="$(hostname):${NUM_TASKS}"
    export MASTER_ADDR=$(hostname)
else
    NUM_TASKS=8
    HOST_CONFIG="g10:8"
    export MASTER_ADDR=$(echo ${HOST_CONFIG} | awk -F: '{print $1}')
fi

RUNNER_CMD="mpirun --prefix $(dirname `which mpirun`)/../ \
    -x MASTER_ADDR -x MASTER_PORT \
    -x LD_LIBRARY_PATH -x PATH \
    -x TRACE_NAME \
    -x NCCL_DEBUG \
    -x NCCL_NET_GDR_LEVEL \
    -x NCCL_DEBUG_SUBSYS \
    -x NCCL_IB_DISABLE \
    -x CLUSTER_NAME \
    -x PLATFORM \
    -x TMPDIR=$PULP_TMPDIR \
    --map-by ppr:8:numa --bind-to core --report-bindings \
    -np $NUM_TASKS --host $HOST_CONFIG"
# if [[ $EXE_MODE == 'GPUS' ]]; then
#     RUNNER_CMD="$RUNNER_CMD \
#         --host $HOST_CONFIG \
#     "
# fi

NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPU_NUM}_$(date "+%Y%m%d-%H%M%S")"
NSIGHT_CMD=""
set -x
# export CUDA_LAUNCH_BLOCKING=1 # for debugging
# export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: important for cc overlap !!!
$NSIGHT_CMD \
$RUNNER_CMD \
./scripts/runtime/bench_dist_attn.sh \
python $PYTHON_EXECUBLE \
    $LOGGING_ARGS \
    
set +x
exit 0


# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS"


set -x
# export TORCH_USE_CUDA_DSA=1 # use it in **compile-time** of pytorch for debugging
# export TORCH_SHOW_CPP_STACKTRACES=1 # for debugging
# export CUDA_LAUNCH_BLOCKING=1 # for debugging
export CUDA_DEVICE_MAX_CONNECTIONS=32    # [NOTE]: important for cc overlap !!!
time \
$RUNNER_CMD \
./scripts/runtime/bench_dist_attn.sh \
python $PYTHON_EXECUBLE \
    $LOGGING_ARGS \
# -c ${CPU_PER_TASK} \  # Don't use it for task1 !!!

set +x
exit 0

set +x


