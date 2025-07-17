#!/bin/bash

export TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

# Cluster-related Envs:
export CLUSTER_NAME=bingxing
export PLATFORM='H800'
PARTITION=H800
# End

source $1   # May overwrite the default settings
mkdir -p logs/${EXP_NAME}

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$NPROC_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
-K \
--cpu-bind none \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi
if [ ! -z "$CPU_PER_TASK" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        --cpus-per-task=$CPU_PER_TASK \
    "
fi

# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS ./scripts/runtime/executor_bx.sh"

set -x
$RUNNER_CMD \
$EXECUTABLE 2>&1 | tee logs/${EXP_NAME}/output_${TIMESTAMP}.log

set +x
