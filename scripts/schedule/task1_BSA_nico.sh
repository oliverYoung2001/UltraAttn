#!/bin/bash

set -x

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')

EXP_NAME='task1_BSA'

mkdir -p results
mkdir -p results/${EXP_NAME}

# Envs:
export CLUSTER_NAME=zhipu_planck
export PLATFORM='H100'
# # set pulp tmp dir; [TODO]: `TMPDIR` is used by both pulp and openmpi
export PULP_TMPDIR=./search_algo/tmp
# export TMPDIR=search_algo/tmp
PYTHON_EXECUBLE=search_algo/task1_bsa.py
LOGGING_ARGS=""
# Envs for gurobipy
export GUROBI_NUM_THREADS=90

time \
python $PYTHON_EXECUBLE \
    $LOGGING_ARGS \
2>&1 | tee results/${EXP_NAME}/output_${TIMESTAMP}.log
        


set +x
