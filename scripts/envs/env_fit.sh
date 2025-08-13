# Spack
source /home/fit/zhaijdyhy/.local/spack/share/spack/setup-env.sh
#   CUDA
spack load cuda@12.8.1
# spack load cuda@12.4.1
# #   CUDNN
# spack load cudnn@9.8.0.87-12
# #   OPENMPI
# spack load openmpi@5.0.7
# export LD_LIBRARY_PATH=$(dirname $(which mpicxx))/../lib:$LD_LIBRARY_PATH
# Numactl
spack load numactl

# conda
source /ssd/yanghy/.local/miniconda3/bin/activate
conda deactivate && conda deactivate && conda deactivate && conda activate ultra_attn

# # set GUROBI licence env
# export GRB_LICENSE_FILE=/home/yhy/mnt/.local/gurobi/gurobi.lic

# [NOTE]: tmux of qy has bug along with conda, thus the solution is copy the $PATH out of tmux into tmux  !!!
export http_proxy=127.0.0.1:18901
export https_proxy=127.0.0.1:18901
