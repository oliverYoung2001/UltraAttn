import argparse
import yaml

def print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ Arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of Arguments ---------------------',
              flush=True)
    
def parse_parallel_args(parser):
    parser.add_argument('--SP', type=int, nargs='+', default=None, help='Dimension of Sequence Parallel')

def parse_shape_args(parser):
    parser.add_argument('--Sq', type=int, default=None, help='Sequence Length of Q')
    parser.add_argument('--Skv', type=int, default=None, help='Sequence Length of KV')
    parser.add_argument('--Nhq', type=int, default=None, help='Number of Heads for Q')
    parser.add_argument('--Nhkv', type=int, default=None, help='Number of Heads for KV')
    parser.add_argument('--BS', type=int, default=None, help='Batch Size')
    parser.add_argument('--D', type=int, default=None, help='Dimension of Embedding')

def parse_pattern_args(parser):
    parser.add_argument('--pattern_type', type=str, default=None, choices=[None, 'causal', 'full', 'star', 'stream', 'stride'], help='Attention Pattern Type')
    parser.add_argument('--pattern_sparsity', type=float, default=None, help='Attention Pattern Sparsity')

def parse_experiment_args(parser):
    parser.add_argument('--transform_mode', type=str, default=None, choices=[None, 'None', 'Greedy', 'bf'], help='Implementation for Parallel Graph Transformation Engine')
    parser.add_argument('--lowering_type', type=str, default=None, choices=[None, 'Flexflow', 'ILP'], help='Implementation for Lowering Engine')
    
def parse_args():
    parser = argparse.ArgumentParser(description='Search Algorithms')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parse_parallel_args(parser)
    parse_shape_args(parser)
    parse_pattern_args(parser)
    parse_experiment_args(parser)
    
    args = parser.parse_args()
    return args

def recursive_update(config, args):
    if isinstance(config, dict):
        for key, value in config.items():
            if hasattr(args.key):
                if getattr(args, key) is None:
                    setattr(args, key, value)
            else:
                recursive_update(value, args)
    elif isinstance(config, list):
        for i, value in enumerate(config):
            recursive_update(value, args)
    
def parse_config():
    args = parse_args()
    assert args.config is not None
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    recursive_update(config, args)
    print_args(args)
    return args
    