#!/bin/bash
REPO_DIR=/ssd/yanghy/llm
REPO_ROOT=$REPO_DIR/UltraAttn

# Clone repo
pushd $REPO_DIR
git clone https://github.com/oliverYoung2001/UltraAttn.git --recurse-submodules
or
git clone https://github.com/oliverYoung2001/UltraAttn.git && cd UltraAttn && git submodule update --init --recursive
# build nccl v2.21.5-1
cd ./UltraAttn/third_party/comm_test/third_party/nccl
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
              -gencode=arch=compute_90,code=sm_90"
make -j src.build NVCC_GENCODE=${NVCC_GENCODE}
popd
# End

cd $REPO_ROOT

# Prepare conda environment
source ./scripts/envs/env_bx.sh
conda create -n ultra_attn python=3.10 -y && conda activate ultra_attn
#   install stable pytorch. Currently v2.7.1+cu128
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#   install other dependencies
pip install -r requirements.txt

# Prepare flash/burst attention
pushd ./third_party/flash-attention
MAX_JOBS=128 pip install -e .   # Firstly run on login node to pip install dependencies. Secondly run on gpu node to compile highly concurrently
popd

# GUROBI free academic license
# On fit
# /home/zhaijidong/yhy/Software/licensetools12.0.0_linux64/grbgetkey 79171a71-14ae-4ab7-9dec-675e7026e2c6
# save GUROBI license in: /home/fit/zhaijd/yhy/.local/gurobi/gurobi.lic
# /home/yhy/mnt/llm/Software/grbgetkey 475d4a64-1498-49d5-affe-2db848705ca0
# save GUROBI license in: /home/yhy/mnt/.local/gurobi/gurobi.lic

# "Times New Roman" font for plot
# sudo apt update
# sudo apt install ttf-mscorefonts-installer -y
# Download 'Times New Roman.ttf' at https://github.com/justrajdeep/fonts/blob/master/Times%20New%20Roman.ttf to ???/Software
mkdir -p ~/.fonts
cp ./plot/fonts/times_new_roman.ttf ~/.fonts
fc-cache -f -v # refresh font cache
fc-list | grep "Times New Roman" # check whether it is installed
rm -r ~/.cache/matplotlib   # remove cache of matplotlib

# Tutorial: How to add a specific branch of the repo to submodule ?
# git submodule add -f -b <branch> <repo> <target_address> # Add submodule, If you find the output of `git submodule` is empty
