import torch
from search_algo.search_engine import Machine_Config

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
    def __init__(self):
        pass
    
    def insert(self):
        pass
    
    def delete(self):
        pass
    
    def update(self):
        pass
    
    def update_m_config(self, m_config: Machine_Config):
        self.m_config = m_config
    
    def read(self):
        pass
