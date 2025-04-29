#!/bin/bash

export CLUSTER_NAME='fit'
export PLATFORM='A800'
export PLATFORM='H800'

export CLUSTER_NAME=hamming
export PLATFORM='H100'
# set pulp tmp dir
export TMPDIR=./search_algo/tmp

# ./scripts/schedule/cpu_task_${CLUSTER_NAME}.sh \
python search_algo/intra_attn_gen.py \
