#!/bin/bash

# Clone repo
pushd ~/yhy/llm
git clone https://github.com/oliverYoung2001/UltraAttn.git --recurse-submodules
or
git clone https://github.com/oliverYoung2001/UltraAttn.git && cd UltraAttn && git submodule update --init --recursive
# build nccl v2.21.5-1
cd ./UltraAttn/third_party/comm_test/third_party/nccl
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
              -gencode=arch=compute_90,code=sm_90"
make -j src.build NVCC_GENCODE="$NVCC_GENCODE"
popd
# End

CPUS_PER_NODE=104

# Install Openmpi
#   1. From source
export OPENMPI_HOME=/home/fit/zhaijd/yhy/.local/openmpi
pushd ~/yhy/Software
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz
tar -zxvf openmpi-4.1.8.tar.gz
cd openmpi-4.1.8
./configure --prefix=$OPENMPI_HOME
make -j $CPUS_PER_NODE all
make install
popd
#   2. From spack
#   TODO
# End

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
# On fit
# /home/zhaijidong/yhy/Software/licensetools12.0.0_linux64/grbgetkey 79171a71-14ae-4ab7-9dec-675e7026e2c6
# save GUROBI license in: /home/fit/zhaijd/yhy/.local/gurobi/gurobi.lic
# /home/yhy/mnt/llm/Software/grbgetkey 475d4a64-1498-49d5-affe-2db848705ca0
# save GUROBI license in: /home/yhy/mnt/.local/gurobi/gurobi.lic

# On qiyuan
# install pytorch nightly for flexattn in yhy_pt_nightly envs
conda activate yhy_pt_nightly
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# On Fit_Cluster
conda activate yhy_ultra
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# "Times New Roman" font for plot
# sudo apt update
# sudo apt install ttf-mscorefonts-installer -y
Download 'Times New Roman.ttf' at https://github.com/justrajdeep/fonts/blob/master/Times%20New%20Roman.ttf to ???/Software
mkdir -p ~/.fonts
cp ???/Software/'Times New Roman.ttf' ~/.fonts
sudo fc-cache -f -v # refresh font cache
fc-list | grep "Times New Roman" # check whether it is installed
rm -r ~/.cache/matplotlib   # remove cache of matplotlib


# Add main branch of UltraAttn_baseline as submodule
git submodule add -f -b main https://github.com/oliverYoung2001/UltraAttn_baseline.git ./third_party/UltraAttn_baseline
