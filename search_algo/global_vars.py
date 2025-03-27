
from enum import Enum
import os

# GLOBAL CONSTANTS
class TASK_STATUS(Enum):
    EMPTY = - 1
    UNSETTLED = - 2

# GLOBAL VARS
BYTE_MULTPLE_UP = 1024
BYTE_MULTPLE_DOWN = 1000
SCHEDULE_UNIQUE_ID = - 1
# DATABASE_ROOT = f'{os.path.dirname(__file__)}/../database'
PROC_INFO: dict

def get_global_var(key: str):
    assert key in globals().keys(), f'Invalid key: {key}'
    return globals()[key]

def set_global_var(key: str, value):
    globals()[key] = value
