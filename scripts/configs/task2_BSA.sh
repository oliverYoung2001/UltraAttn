CLUSTER_NAME=${CLUSTER_NAME:-UNKNOWN_CLUSTER}
EXP_NAME=task2_BSA_${CLUSTER_NAME}_$2
PYTHON_EXECUBLE=search_algo/task2_bsa.py

# Parallel Parameters:
GPUS_PER_NODE=8
NPROC_PER_NODE=$GPUS_PER_NODE
NNODES=2
HOST='g[0002,0004]'
HOST='g[0004,0006]'
# HOST='g[0006,0009]'
# HOST='g[0004,0028]'
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
CPUS=128
CPU_PER_TASK=$(( CPUS / NPROC_PER_NODE ))   # [NOTE]: Unnecessary for performance.

# Attention Patterns
MODEL_ARGS=()
if [ ! -z $2 ]; then
    MODEL_ARGS+=(
        --exp-class $2 # ['bsa_train', 'dense_train', 'bsa_infer']
    )
fi

# Profiling switch
USE_TORCH_PROFILE=False
USE_NSIGHT=False


# Profiling ARGS:
export TRACE_NAME=${CLUSTER_NAME}
#   Torch profile
if [ $USE_TORCH_PROFILE == "True" ]; then
    TB_DIR=./prof_results/tb
    mkdir -p $TB_DIR
    LOGGING_ARGS=""
fi
# Nsight
if [ $USE_NSIGHT == "True" ]; then
    NSYS_DIR=./prof_results/nsys_orchestrate
    mkdir -p $NSYS_DIR
    NSIGHT_CMD="nsys profile --mpi-impl=openmpi -o ${NSYS_DIR}/${TRACE_NAME}_w${GPU_NUM}_$(date "+%Y%m%d-%H%M%S")"
    NSIGHT_CMD=""
fi
# End

# Other envs:
export GUROBI_NUM_THREADS=128

# Execution Scripts:
EXECUTABLE="python $PYTHON_EXECUBLE \
    $LOGGING_ARGS \
    ${MODEL_ARGS[@]} \
"
# EXECUTABLE="./tmp/test_bind_core.sh"
# EXECUTABLE="python ./tmp/test_bind_core.py"
