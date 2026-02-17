# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VLIF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--pid', '-p', type=bool, default=False, help='whether to use PID module')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id to use')

    args, _ = parser.parse_known_args()

    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.0001],
        'n_layers': [2],
        'gpu_id': args.gpu,
        'pid': args.pid
    }


    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


