import argparse
import glob
import os
import sys
import time
import ipdb
from collections import defaultdict
from srblib import abs_path
import copy
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
from my_utils import mkdir_p
import warnings
warnings.filterwarnings("ignore")

## TODO: Fine-tuning from checkpoint, Learning Rate Scheduler (based on what was used in the paper),

import settings
from regression_model_and_dataset import BertForRegression, LoggingCallback

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for fine-tuning on synthetic dataset')

    ## Program Arguments
    parser.add_argument('--seed', type=int, help='Seed value. Default=0', default=0)

    ##########################
    ## NOTE: This will go first
    # add model specific args
    parser = BertForRegression.add_model_and_data_specific_args(parser, os.getcwd())

    ## Trainer Arguments
    parser = pl.Trainer.add_argparse_args(parser)



    ## Parse the arguments
    return parser.parse_args()


def main(args):

    ## Debugging Mode
    if sys.gettrace() is not None:
        print('In debug mode')
        args.profiler = 'simple'
        args.num_workers = 0
        args.pin_memory=False
        args.default_root_dir = mkdir_p(abs_path('./tmp_dir'))
        args.stochastic_weight_avg = True
        args.limit_train_batches = 0.001
        args.limit_val_batches = 0.001
        args.num_sanity_val_steps = 0
        args.terminate_on_nan = True
        args.deterministic = True
        args.auto_select_gpus = False
        args.fast_dev_run = False # Quick Check Working
        args.progress_bar_refresh_rate = 0
        args.gpus = 1
        args.precision=16
        args.train_batch_size=32
        # args.val_batch_size=4
        args.freeze_encoder = True
        args.verbose = True
        args.max_epochs = 2

    # ipdb.set_trace()
    ## Model
    dict_vars = vars(args)
    model = BertForRegression(**dict_vars)
    print(f'Model Init Done')

    ## TODO: Early Stopping Callback
    ## Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch:02d}-{val_loss:.5f}-{step_count}',
                                                       dirpath=None,
                                                       prefix="best_model_checkpoint",
                                                       monitor="val_loss",
                                                       mode="min",
                                                       save_top_k=args.save_top_k,
                                                       )
    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[LoggingCallback(), checkpoint_callback],
                                            )

    trainer.fit(model)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f'\nBEST MODEL PATH IS {best_model_path}')
    aa = 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s_time = time.time()
    f_time = ''.join(str(s_time).split('.'))
    print(f'fixed time is {f_time}')

    args = get_arguments()
    pl.seed_everything(args.seed)

    main(args)

