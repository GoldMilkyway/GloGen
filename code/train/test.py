#%%
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

from time import time, ctime
from omegaconf import OmegaConf
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

def seed_everything(SEED):
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    pl.utilities.seed.seed_everything(seed=SEED)
    torch.backends.cudnn.deterministic = True ###
    torch.backends.cudnn.benchmark = False ###

def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file")
    parser.add_argument("--group_avg", action="store_true")
    parser.add_argument("--method", choices=["original", "prompt_global", "prompt_gen",
                                             "prompt_glogen", "prompt_lowgen", "prompt_lowglogen"], required=True)
    parser.add_argument("--backbone", choices=["resnet1d", "mlpbp", "spectroresnet"], required=True)
    parser.add_argument("--gen_ip", action="store_true")
    parser.add_argument("--load_path", type=str, default=None, help="Test model Path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser

def parser_to_config(parser, config):
    """
    Add parser argument into oemgaconfig
    """
    args = vars(parser.parse_args())
    for key, item in args.items():
        config[key] = item
        if key == "lr" or key == "wd":
            config.param_model[key] = item
        if key == "load_path":
            config.param_test.model_path = item
    return config

def main(args):        
    if os.path.exists(args.config_file) == False:         
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()
    config = OmegaConf.load(args.config_file)
    config = parser_to_config(parser, config)

    # set seed
    seed_everything(config.seed)

    if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
        solver = solver_s2s(config)
    elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp']:
        solver = solver_s2l(config)

    solver.test()
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    # =============================================================================
    # output
    # =============================================================================


#%%
if __name__=='__main__':
    parser = get_parser()
    main(parser.parse_args())