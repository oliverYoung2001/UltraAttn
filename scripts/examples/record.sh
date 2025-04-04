
# Cluster Profiling [TODO]
pushd ~/yhy/llm/UltraAttn
#   comp profiling
pushd third_party/kernel_profiler
./scripts/bench_ops_m2_py.sh    # Output is located in ./prof_data/tmp; Move it to UltraAttn/prof_data/${CLUSTER_NAME}
popd
#   comm profiling
pushd third_party/comm_test
./scripts/wrapper_conflict_bench_zhipu.sh 2>&1 | tee ./prof_data/zhipu_???/cb_<CP>_<nodes>.log
popd
popd




export PLATFORM='A800'
export PLATFORM='H800'

# 🌟 For Full/Causal
# 保留，独立的，提前做好
# Generate the intra-attention w/o causal results
# For intra-attn, full, non fused, manually cc schedule
# NUM_SCHEDULES=(logN + 1)
# results: execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP4_fob=0/1
./scripts/intra_attn_gen.sh 2>&1 | tee ./results/search_intra_SP=4_noncausal_all.log
# On Fit
# SP = 1, 2; SP = 1, 4; SP = 1, 8
./scripts/schedule/intra_attn_gen.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=2_noncausal_all.log
./scripts/schedule/intra_attn_gen.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=4_noncausal_all.log
./scripts/schedule/intra_attn_gen.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=8_noncausal_all.log
✅✅

# 保留，独立的，提前做好
# Generate the intra-attention w causal results
# results: execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP4_fob=0/1_causal=True
# NUM_SCHEDULES=2x2 (Raw, Fused) x (ILP, Flexflow)
# [NOTE]: Use GUROBI as backend of pulp !!!
./scripts/search_engine_qy.sh 2>&1 | tee ./results/search_intra_SP=4_causal_all.log
# use generate_intra_execution_plans
# On Fit
# SP = 1, 4; # SP = 1, 8 -> Useless !!!
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=4_causal_all_gurobi.log
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=8_causal_all_gurobi.log
✅✅

# [TODO]: 想办法去掉！！！
# # Generate the inter-attention SP0 = 1 before profile !!! [WORKAROUND] for profile intra-attn
# Inter: (1, 8)
# [HACK] INTER_COMP_FILE_NAME with oldones: 'prof_data/qiyuan/wrapper_intra_SP=8_all.log'
# Both run with causal=True and causal=False
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_inter_SP=1_causal_all.log
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_inter_SP=1_full_all.log
✅✅

# 保留，独立的，提前做好
# Generate profile file: wrapper_intra_SP=8_all.log
./scripts/wrapper_qy.sh 2>&1 | tee ./prof_data/wrapper_intra_SP=8_all.log
# use mode='profile'
# On Fit, # use profile_all_intra_attn
# For A800
./scripts/runtime/wrapper.sh 2>&1 | tee ./prof_data/fit/wrapper_intra_SP=8_all.log
# About 9 hours on A800 [TODO]: accelerate it !!!
# For H800
./scripts/runtime/wrapper.sh 2>&1 | tee ./prof_data/fit/wrapper_intra_SP=8_all_H800.log
# About 4 hours on H800 with parallel [TODO]: accelerate it !!!
# But need "inter_SP1_fob=x" schedules !!!
✅✅

# 整合入task1 (3)
# Generate the inter-attention w causal results
# results: inter_SP8_fob=0/1
./scripts/search_engine_qy.sh 2>&1 | tee ./results/search_inter_SP=4_causal_all.log
# generate_inter_execution_plans
# On Fit
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_inter_SP=4_causal_all.log
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_inter_SP=8_causal_all.log
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_inter_SP=2_causal_all.log
# [TODO]: more fine-grained division for SP=(2, 8)
# SP=(2, 8): no fused ?
✅✅

# Generate the inter-attention w/o causal results   # No !!!
# None

# task2
# Run the inter-attention w&w/o causal results
./scripts/wrapper_qy.sh 2>&1 | tee ./results_exp/wrapper_inter_SP=4,8_causal=True.log
# use mode='test'
# On Fit
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=2,8_causal=True.log
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=4,8_causal=True.log
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=8,8_causal=True.log
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=2,8_causal=False.log
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=4,8_causal=False.log
./scripts/runtime/wrapper.sh 2>&1 | tee ./results_exp/fit/${PLATFORM}/wrapper_inter_SP=8,8_causal=False.log
✅





# 🌟 For Block Sparse Attention
# [Task1]
# task1 part0: top->down
./scripts/schedule/task1_BSA.sh 2>&1 | tee ./results/task1_BSA.log

# 融合入task1 part1
# Step 1: Generate the intra-BSA 
# results: execution_plans/{CLUSTER_NAME}/{PLATFORM}/intra_SP4_fob=0/1_{BSA_PATTERN}
# NUM_SCHEDULES=2x2 (Raw, Fused) x (ILP, Flexflow)
# [NOTE]: Use GUROBI as backend of pulp !!!
./scripts/search_engine_qy.sh 2>&1 | tee ./results/search_intra_SP=8_BSA_all.log
# use generate_intra_execution_plans
# On Fit
# SP = 1, 4; # SP = 1, 8 -> Useless !!!
# ./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=4_BSA_all.log
./scripts/schedule/search_engine_old.sh 2>&1 | tee ./results/schedule/${PLATFORM}/search_intra_SP=8_BSA_all.log
# [NOTE]: No Transformations Selected !!!
✅

# 融合入task1 part2
# Step 2: Profile all BSA at intra_SP=8: wrapper_intra_SP=8_all.log
# Use profile_all_intra_attn
# For A800
./scripts/runtime/wrapper.sh 2>&1 | tee ./prof_data/fit/wrapper_intra_SP=8_all_BSA.log
# About 9 hours on A800 [TODO]: accelerate it !!!
# For H800
./scripts/runtime/wrapper.sh 2>&1 | tee ./prof_data/fit/wrapper_intra_SP=8_all_H800.log
# About 4 hours on H800 with parallel [TODO]: accelerate it !!!
# But need "inter_SP1_fob=x" schedules !!!

# 融合入task1 part3
# Step3: Generate execution plans for all BSA at inter_SP=2,4,8: [TODO]

# [Task2]
# Step4: Profile all BSA at inter_SP=2,4,8: [TODO]

# [END]


# # End to End: HACK

# [Dreprecate]: begin
# # Ablation 1: Workload Allocation. Searching results vs expert-designed results


# # Ablation 2: Non-fused vs min(Non-fused, Fused)
# # parse "wrapper_intra_SP=8_all.log", Nh=1, noncausal, SP=(1,8), Sg=..., Fob=0/1, max(non-fused) vs max(all)

# # Ablation 3: ILP vs Flexflow
# # parse "SP=1,4_Sg=1k_causal_ablation01.log" or "SP=1,4_Sg=1k_causal_ablation1.log"

# # Searching of Computation Workload Allocation Engine:
# ./scripts/search_engine_qy.sh 2>&1 | tee ./results_exp/search_engine_qy_N=?_locality_fob=?.log

# srun -p arch -N 4 --ntasks-per-node=8 --gres=gpu:8 --mem 256G -K  -c 16 hostname
# [Dreprecate]: end

