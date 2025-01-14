#!/bin/bash

CLUSTER_NAME=fit
NNODES=1
NTASKS_PER_NODE=1
PARTITION=a01
HOST="g01"
HOST="g07"
# PARTITION=h01
# HOST="g44"

HOST=None

export MASTER_PORT=$((RANDOM % 12000 + 10000))

MEM_PER_CPU=256G
# --mem-per-cpu $MEM_PER_CPU \
MEM_PER_NODE=256G

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$NTASKS_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS"

set -x
$RUNNER_CMD \
$@

set +x
