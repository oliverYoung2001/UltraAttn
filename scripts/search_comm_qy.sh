#!/bin/bash

# set pulp tmp dir
export TMPDIR=./search_algo/tmp

./scripts/cpu_task_qy.sh \
python search_algo/search_comm.py \