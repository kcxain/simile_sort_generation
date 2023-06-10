"""Loads and pre-processes the original dialog data."""

import os
import argparse
import importlib
from processor_SPGC_mlr import SimileMLRLossProcessor
# ==================================================================
# Constants
# ==================================================================
ALL_SIMILE_TYPES = [
    'positive',                 # 0
    'vehicle_isa_replace',      # 1
    'single_random_replace',    # 2
    # 'property_random_replace',
    'both_random_replace'       # 3
]


# ==================================================================
# Functions
# ==================================================================
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--src_dataset', '-sd',
        choices=['SPGC'],
        default='SPGC',
        help='Source raw dataset name.')
    parser.add_argument(
        '--tgt_dataset', '-td',
        choices=['SPGC_mlr'],
        default='SPGC_mlr',
        help='Target processed dataset name.')
    parser.add_argument(
        '--simile_types', '-rt',
        choices=['1+2+3+4', '1+2+3', '1+3'],
        default='1+2+3+4',
        help='Desired simile types.')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the simile sentence.')

    args = parser.parse_args()
    return args


def get_response_types(response_types_str):
    response_types = []
    for type_str in response_types_str.split('+'):
        type_idx = int(type_str) - 1
        response_type = ALL_SIMILE_TYPES[type_idx]
        response_types.append(response_type)
    return response_types


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    args = parse_opt()
    input_dir_path = os.path.join('./dataset', args.src_dataset)
    output_dir_path = os.path.join('../data', args.tgt_dataset)
    response_types = get_response_types(args.response_types)
    processor = SimileMLRLossProcessor(
        input_dir_path=input_dir_path,
        output_dir_path=output_dir_path,
        simile_types=response_types,
        max_seq_length=args.max_seq_length)
    processor.prepare()
    # processor.analyze()
