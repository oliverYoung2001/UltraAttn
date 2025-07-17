run_figure7:
	time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh bsa_train       # About 5 hours
	time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task2_BSA.sh bsa_train		# About 21mins(8/16) + ???(32/64)
	mv database database_bsa_train

gen_figure7:
	python plot/da_bsa_training_pick.py

run_figure8:
	time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh dense_train     # About 8 hours
	time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task2_BSA.sh dense_train		# About 21mins(8/16) + ???(32/64)
	mv database database_dense_train

gen_figure8:
	python plot/da_dense_training_pick.py

run_figure9:
	time ./scripts/runtime/wrapper_bx.sh ./scripts/configs/task1_BSA.sh bsa_infer       # About 10 minutes
	mv database database_bsa_infer

gen_figure9:
	python plot/da_bsa_inference_pick.py

figure6: run_figure7 run_figure8 run_figure9
	python plot/e2e_pick.py

figure7:	# About 6 hours
	$(MAKE) run_figure7
	$(MAKE) gen_figure7

figure8:	# About 9 hours
	$(MAKE) run_figure8
	$(MAKE) gen_figure8

figure9:	# About 10 minutes
	$(MAKE) run_figure9
	$(MAKE) gen_figure9

figure10:
	python plot/strong_scalability_training_pick.py
	python plot/strong_scalability_inference_pick.py

figure11:
	python plot/sim_accuracy.py

spack_packages:
	# Download spack
	mkdir ~/.local
	git clone https://github.com/spack/spack.git ~/.local/spack
	# Activate spack
	source ~/.local/spack/share/spack/setup-env.sh
	# Install spack packages
	spack install cuda@12.8.1
	spack install numactl@2.0.18
	# Activate spack packages
	spack load cuda@12.8.1
	spack load numactl@2.0.18

install_miniconda:
	mkdir -p ./tmp
	mkdir -p ~/.local
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./tmp/miniconda.sh
	bash ./tmp/miniconda.sh -b -u -p ~/.local/miniconda3
	# Init
	source ~/.local/miniconda3/bin/activate

prepare_conda_env: install_miniconda
	conda create -n ultra_attn python=3.10 -y
	conda activate ultra_attn
	# install dependencies
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt


compile:
	# build nccl v2.21.5-1
	pushd ./third_party/comm_test/third_party/nccl
	NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
				-gencode=arch=compute_90,code=sm_90"
	make -j src.build NVCC_GENCODE=${NVCC_GENCODE}
	popd
	# Prepare flash/burst attention
	pushd ./third_party/flash-attention
	MAX_JOBS=128 pip install -e .   # Firstly run on login node to pip install dependencies. Secondly run on gpu node to compile highly concurrently
	popd

cluster_profile:
	#   comp profiling
	pushd third_party/kernel_profiler
	OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/configs/bench_ops_m2_py.sh
	popd
	#   comm profiling
	pushd third_party/comm_test
	OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/task_configs/cb_ultra_8.sh
	OUTPUT_DIR=../../logs/m_configs time ./scripts/wrapper_bx.sh ./scripts/task_configs/cb_ultra_16.sh
	popd
