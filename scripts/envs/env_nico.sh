source ~/env.sh
# spack load gcc@10.2.0
# spack load cuda@11.8
spack load gcc@11.2.0%gcc@=12.2.0
# spack load cudnn@8.8.0.121-11.8
spack load cudnn@8.8.0.121-12.0
spack load cuda@12.1.1
# conda activate vllm
conda deactivate && conda deactivate && conda deactivate
conda activate mg

# set GUROBI licence env
export GRB_LICENSE_FILE=/home/yhy/.local/gurobi/gurobi.lic

# Openmpi
export OPENMPI_HOME=/home/yhy/.local/openmpi-4.1.6
export PATH="$OPENMPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

export C_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$C_INCLUDE_PATH"  # for #include <mpi.h>
export CPLUS_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$CPLUS_INCLUDE_PATH"  # for #include <mpi.h>
export LD_LIBRARY_PATH="$(dirname `which nvcc`)/../lib64:$LD_LIBRARY_PATH"  # for -lcudart

# export http_proxy="http://127.0.0.1:8901"
# export https_proxy="http://127.0.0.1:8901"

# export http_proxy="http://nico0:8901"
# export https_proxy="http://nico0:8901"

# unset http_proxy
# unset https_proxy

# nico0: 127.23.18.10
# nico2: 127.23.18.2

