CLUSTER_NAME=bingxing
PLATFORM=H800
# mkdir -p database/$CLUSTER_NAME/$PLATFORM/m_configs
mkdir -p logs/m_configs

# Cluster Profiling [TODO]
#   comp profiling
pushd third_party/kernel_profiler
# OUTPUT_DIR=../../database/bingxing/H800/m_configs ./scripts/wrapper_bx.sh ./scripts/configs/bench_ops_m2_py.sh
OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/configs/bench_ops_m2_py.sh
popd

#   comm profiling
pushd third_party/comm_test
OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/task_configs/cb_ultra_8.sh
OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/task_configs/cb_ultra_16.sh
# ./scripts/wrapper_conflict_bench_bx.sh 2>&1 | tee ../../database/bingxing/H800/m_configs/cb_test_8.log
# ./scripts/wrapper_conflict_bench_bx.sh 16 2>&1 | tee ../../database/bingxing/H800/m_configs/cb_test_16.log
popd

# pushd ~/yhy/llm/UltraAttn
# #   comp profiling
# pushd third_party/kernel_profiler
# ./scripts/bench_ops_m2_py.sh    # Output is located in ./prof_data/tmp; Move it to UltraAttn/prof_data/${CLUSTER_NAME}
# popd
# #   comm profiling
# pushd third_party/comm_test
# ./scripts/wrapper_conflict_bench_hamming.sh 8 2>&1 | tee ./prof_data/hamming/cb_8.log
# ./scripts/wrapper_conflict_bench_hamming.sh 16 2>&1 | tee ./prof_data/hamming/cb_16.log
# popd
# popd


# export PLATFORM='A800'
# export PLATFORM='H800'

# Task1: step1~3
time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh bsa_train       # About 5 hours
time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh dense_train     # About 8 hours
time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh bsa_infer       # About 10 minutes

# Task2: step4
time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task2_BSA.sh bsa_train       # About 21mins(8/16) + ???(32/64)
time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task2_BSA.sh dense_train
# time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task2_BSA.sh bsa_infer
