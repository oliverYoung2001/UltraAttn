lscpu
nvidia-smi topo -m
ibstatus
lspci -tvv
lstopo --output-format svg <pic_name>.svg
# install lstopo with conda
conda install -c conda-forge libhwloc

# attention on single gpu (Flexattn, flashattn)
from qiyuan:~/yhy/llm/megatron-mg
# comm intra-node/inter-node
from qiyuan:~/yhy/llm/comm_test
