import os
import torch
from search_algo.utils import convert_profile_data_to_map, FinitePriorityQueue, \
                              convert_profile_data_to_comm_map, convert_node_profile_data_to_comp_map, \
                              find_file_with_regex
import json
import shutil

class Prof_DB():
    # 1. Base profile for cluster
    #   2 x 2
    #   m_config
    # 2. Intra-node execution plans. (BSA_Configs) x (f/b) x (Ss) x (Nhs) x (Ablations(2x2))
    #   KV pairs
    # 3. Intra-node execution times. (BSA_Configs) x (f/b) x (Ss) x (Nhs) x (Ablations(2x2))
    #   KV pairs
    # 4. Inter-node execution plans. 
    #   KV pairs
    # 5. Inter-node execution times. 
    #   KV pairs
    def __init__(self, CLUSTER_NAME, PLATFORM, gloo_global_group):
        # CLUSTER_NAME, PLATFORM = os.environ.get('CLUSTER_NAME', None), os.environ.get('PLATFORM', None)
        self.CLUSTER_NAME = CLUSTER_NAME
        self.PLATFORM = PLATFORM
        self.DATABASE_ROOT = f'{os.path.dirname(__file__)}/../database/{CLUSTER_NAME}/{PLATFORM}'
        self.M_CONFIG_DIR = f'{self.DATABASE_ROOT}/m_configs'
        self.RAW_M_CONFIG_DIR = f'logs/m_configs'
        
        self.INTRA_BSA_ALLOCATION = f'{self.DATABASE_ROOT}/intra_bsa_allocation.json'
        self.INTRA_BSA_EXE_PLANS_DIR = f'{self.DATABASE_ROOT}/intra_bsa_exe_plans'
        self.INTRA_BSA_EXE_PLANS_KV = f'{self.DATABASE_ROOT}/intra_bsa_exe_plans_kv.json'
        self.INTRA_BSA_EXE_PLANS_PROFILE = f'{self.DATABASE_ROOT}/intra_bsa_exe_plans_profile.json'
        
        self.INTER_BSA_ALLOCATION = f'{self.DATABASE_ROOT}/inter_bsa_allocation.json'
        self.INTER_BSA_EXE_PLANS_DIR = f'{self.DATABASE_ROOT}/inter_bsa_exe_plans'
        self.INTER_BSA_EXE_PLANS_KV = f'{self.DATABASE_ROOT}/inter_bsa_exe_plans_kv.json'
        self.INTER_BSA_EXE_PLANS_PROFILE = f'{self.DATABASE_ROOT}/inter_bsa_exe_plans_profile.json'
        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:   # Initialize all above
            os.makedirs(self.DATABASE_ROOT, exist_ok=True)
            if not os.path.isdir(self.M_CONFIG_DIR):
                shutil.copytree(self.RAW_M_CONFIG_DIR, self.M_CONFIG_DIR)
            if not os.path.exists(self.INTRA_BSA_ALLOCATION):
                with open(self.INTRA_BSA_ALLOCATION, 'w') as f:
                    json.dump({}, f)
            os.makedirs(self.INTRA_BSA_EXE_PLANS_DIR, exist_ok=True)
            if not os.path.exists(self.INTRA_BSA_EXE_PLANS_KV):
                with open(self.INTRA_BSA_EXE_PLANS_KV, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(self.INTRA_BSA_EXE_PLANS_PROFILE):
                with open(self.INTRA_BSA_EXE_PLANS_PROFILE, 'w') as f:
                    json.dump({}, f)
            
            if not os.path.exists(self.INTER_BSA_ALLOCATION):
                with open(self.INTER_BSA_ALLOCATION, 'w') as f:
                    json.dump({}, f)
            os.makedirs(self.INTER_BSA_EXE_PLANS_DIR, exist_ok=True)
            if not os.path.exists(self.INTER_BSA_EXE_PLANS_KV):
                with open(self.INTER_BSA_EXE_PLANS_KV, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(self.INTER_BSA_EXE_PLANS_PROFILE):
                with open(self.INTER_BSA_EXE_PLANS_PROFILE, 'w') as f:
                    json.dump({}, f)
        if torch.distributed.is_initialized():
            torch.distributed.barrier(gloo_global_group)
        
        self.m_config = self.create_m_config()
    
    def create_m_config(self):
        PROFILE_FILE_NAME = find_file_with_regex(self.M_CONFIG_DIR, r'^time.*\.json$')[0]
        with open(f'{self.M_CONFIG_DIR}/{PROFILE_FILE_NAME}', 'r') as f:
            profile_data = json.load(f)
        INTER_COMM_FILE_NAME = self.M_CONFIG_DIR + '/' + find_file_with_regex(self.M_CONFIG_DIR, r'^cb_16_.*\.log$')[0]
        INTRA_COMM_FILE_NAME = self.M_CONFIG_DIR + '/' + find_file_with_regex(self.M_CONFIG_DIR, r'^cb_8_.*\.log$')[0]
        from search_algo.search_engine import Machine_Config
        m_config = Machine_Config(None, convert_profile_data_to_map(profile_data['flash_attn']), \
            None,
            convert_profile_data_to_comm_map(INTER_COMM_FILE_NAME, 16),
            convert_profile_data_to_comm_map(INTRA_COMM_FILE_NAME, 1),
        )
        return m_config
    
    def m_config_update_inter_bsa_profile(self):
        with open(self.INTRA_BSA_EXE_PLANS_PROFILE, 'r') as f:
            intra_bsa_exe_plans_profile = json.load(f)
        self.m_config.update_inter_bsa_profile(intra_bsa_exe_plans_profile)

    def insert(self):
        pass
    
    def delete(self):
        pass
    
    def update(self):
        pass
    
    # def update_m_config(self, m_config: Machine_Config):
    #     self.m_config = m_config
    
    def read(self):
        pass
