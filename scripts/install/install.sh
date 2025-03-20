#!/bin/bash

CPUS_PER_NODE=104

# Cluster Profiling
pushd ~/yhy/llm
#   comp profiling

#   comm profiling
git clone 

popd

# Install Openmpi
export OPENMPI_HOME=/home/fit/zhaijd/yhy/.local/openmpi
pushd ~/yhy/Software
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz
tar -zxvf openmpi-4.1.8.tar.gz
cd openmpi-4.1.8
./configure --prefix=$OPENMPI_HOME
make -j $CPUS_PER_NODE all
make install
popd

# For qiyuan:
source ./scripts/env_qy.sh
conda create -n yhy_easycontext python=3.10 -y && conda activate yhy_easycontext
# pip install --pre torch==2.4.0.dev20240324  --index-url https://download.pytorch.org/whl/nightly/cu118
# pip install --pre torch==2.4.0.dev20240324  --index-url https://download.pytorch.org/whl/nightly/cu118
pip install --pre torch==2.1.2  --index-url https://download.pytorch.org/whl/nightly/cu118
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt

# Burst Attention   # [TODO]: use it as a patch
cd ~/yhy/source_codes
git clone https://github.com/oliverYoung2001/flash-attention.git --recurse-submodules && cd flash-attention
# or
git clone <repo_name> && cd flash-attention && git submodule update --init --recursive
# csrc/cutlass use commit: bbe579a
# git submodule add <branch!!!> https://github.com/NVIDIA/cutlass.git csrc/cutlass # Add submodule, If you find the output of `git submodule` is empty
git checkout v2.5.7_burst
# build and install flash-attention
salloc -p a01 -w g11 -N 1 --gres=gpu:8 --cpus-per-task=104 --exclusive
ssh g01
MAX_JOBS=104 pip install -e . # Firstly run on login node to pip install dependencies. Secondly run on gpu node to compile highly concurrently
exit
# install flash-attention
popd

# GUROBI free academic license
# /home/zhaijidong/yhy/Software/licensetools12.0.0_linux64/grbgetkey 79171a71-14ae-4ab7-9dec-675e7026e2c6
# pip install gurobipy
# save GUROBI license in: /home/fit/zhaijd/yhy/.local/gurobi/gurobi.lic

# On qiyuan
# install pytorch nightly for flexattn in yhy_pt_nightly envs
conda activate yhy_pt_nightly
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# On Fit_Cluster
conda activate yhy_ultra
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
