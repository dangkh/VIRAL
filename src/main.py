# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def parse_overrides(tokens):
    """
    Extra command-line arguments override any config key (highest priority):
        --fuse=vpid --vpid_weight=0.5 --vpid_use_id_branch=False --seed=[999,1,2]
    Values are parsed as YAML (numbers, booleans, lists, strings).
    """
    overrides, i = {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith('--'):
            raise ValueError('Unexpected argument: {}'.format(tok))
        tok = tok.lstrip('-')
        if '=' in tok:
            key, raw = tok.split('=', 1)
        else:
            key = tok
            if i + 1 >= len(tokens) or tokens[i + 1].startswith('--'):
                raise ValueError('Missing value for --{}'.format(key))
            i += 1
            raw = tokens[i]
        overrides[key] = yaml.safe_load(raw)
        i += 1
    return overrides


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VLIF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.0001],
        'gpu_id': 0,
    }

    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    if overrides:
        print('Config overrides from command line:', overrides)
    config_dict.update(overrides)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


