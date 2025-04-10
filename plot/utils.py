import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

COLOR_DEF = [
  '#f4b183',
  '#c5e0b4',
  '#ffd966',
  '#bdd7ee',
  "#8dd3c7",
  "#bebada",
  "#fb8072",
]

MARKER_DEF = [
  'o', # ●	圆形
  '^',	# ▲	上三角
  'v',	# ▼	下三角
  's',	# ■	正方形
  'D',	# ◆	菱形
  'x',	# ×	交叉
  '+',	# +	加号
  '*',	# ✱	星号
  'p',	# ⬟	五边形
  'h',  # ⬢	六边形
]


HATCH_DEF = [
  '',
  '////',
  '\\\\\\\\',
  '..',
  'xx',
  '++',
  'oo',
]

COLOR_DEF_LINE = [
  '#f4684d',
  '#ffd93c',
  '#73e069',
  '#6fd7ee',
  '#52d375',
  '#6f6dda',
  '#fb4b43',
  '#4b68d3',
  '#fd6931',
  '#cccccc',
  '#fccde5',
  '#69de3d',
  '#ffd917',
  '#fc522c',
  '#4463cf',
  '#3c7260',
  '#f45e21',
  '#ffc91b',
  '#467447'
]

PLOT_DIR = './figs/'
LOG_DIR = './logs/'

# OUR_SYS = 'TA'
OUR_SYS = 'FlashTensor'

SYS_NAME = {
  'dynamo': 'TorchInductor',
  'torch': 'PyTorch',
  'tensorrt': 'TensorRT',
  'tvm': 'TVM',
  'xla': 'XLA',
  'korch': 'Korch',
  'einnet': 'EinNet',
  'flashattn': 'FlashAttention-2',
  'flashinfer': 'Flashinfer',
  'our': f'{OUR_SYS}',
}

MODEL_NAME = {
  'h2o': 'H$_2$O',
  'roco': 'RoCo',
  'keyformer': 'Keyformer',
  'snapkv': 'SnapKV',
  'corm': 'Corm',
  'attn': 'Vanilla Attention',
  'llama2': 'Llama2',
  'gemma2': 'Gemma2',
}


def parse_csv(path, sep='\t'):
    import pandas as pd
    data = pd.read_csv(path, sep=sep)
    data = data.set_index(data.iloc[:, 0])
    data = data.drop(columns=data.columns[0])
    data.index.name = ''
    data.columns.name = ''
    return data
