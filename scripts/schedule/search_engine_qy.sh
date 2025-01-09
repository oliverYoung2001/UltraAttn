#!/bin/bash

# set pulp tmp dir
export TMPDIR=./search_algo/tmp

# Default Config
CONFIG_FILE='configs/star.yaml'

# Parallel Args
SP="4 1"
PARALLEL_ARGS="
--SP $SP \
"

# Shape Args
Sq=32768
Skv=32768
Nhq=32
Nhkv=32
BS=1
D=128
SHAPE_ARGS="
--Sq $Sq \
--Skv $Skv \
--Nhq $Nhq \
--Nhkv $Nhkv \
--BS $BS \
--D $D \
"

# Pattern Args
PARTTERN_TYPE='star'
SPARSITY=0.25   # 1/4
PATTERN_ARGS="
--pattern_type $PARTTERN_TYPE \
--pattern_sparsity $SPARSITY \
"

# Experiment Args
TRANSFORM_MODE='None'
TRANSFORM_MODE='Greedy'
# TRANSFORM_MODE='bf'
LOWERING_MODE='Flexflow'
LOWERING_MODE='ILP'
EXP_ARGS="
--transform_mode $TRANSFORM_MODE \
--lowering_mode $LOWERING_MODE \
"



./scripts/cpu_task_qy.sh \
python search_algo/main.py \
--config $CONFIG_FILE \
$PARALLEL_ARGS \
$SHAPE_ARGS \
$PATTERN_ARGS \
$EXP_ARGS \
