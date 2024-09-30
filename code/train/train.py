import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

# Load modules
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l
from core.solver_f2l import SolverF2l as solver_f2l
from core.utils import log_params_mlflow, init_mlflow, save_result, max_gpu_memory

from omegaconf import OmegaConf
from time import time, ctime
import mlflow as mf
from shutil import rmtree
from pathlib import Path

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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--global_coeff", type=float, default=1.0)
    parser.add_argument("--gen_coeff", type=float, default=1.0)
    parser.add_argument("--method", choices=["original", "prompt_global", "prompt_gen",
                                             "prompt_glogen"], required=True)
    parser.add_argument("--root_dir", default="/bp_benchmark/")
    parser.add_argument("--result_dirname", default="results")
    parser.add_argument("--layer_num", default=0, type=int)
    parser.add_argument("--glonorm", action="store_true")
    parser.add_argument("--gennorm", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--var", type=float, default=0.0)
    parser.add_argument("--gvar", action="store_true")
    parser.add_argument("--ssvar", type=float, default=0)
    parser.add_argument("--backbone", choices=["resnet1d", "mlpbp", "spectroresnet"], required=True)
    parser.add_argument("--update_encoder", action="store_true")
    parser.add_argument("--update_regressor", action="store_true")
    parser.add_argument("--trigger", action="store_true")
    
    #Quant
    parser.add_argument("--gen_ip", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--count_group", action="store_true", help="Output Counts of group in each fold")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    # Model Selection
    parser.add_argument("--group_avg", action='store_true', default=False)
    parser.add_argument("--loss_selection", action='store_true', default=False)
    parser.add_argument("--worst", action='store_true', default=False)

    # Few-shot Transfer
    parser.add_argument("--shots", default=0, type=int, help="Few-shot Regression")
    parser.add_argument("--transfer", default=None, type=str, choices=["ppgbp", "sensors", "uci2", "bcg", "mimic", "vitaldb"])
    parser.add_argument("--lp", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--create_shot_loader", action="store_true", default=False)

    # Unsupervised Encoder
    parser.add_argument("--pca_encoding", action="store_true")
    parser.add_argument("--n_components", default=64)
    parser.add_argument("--fft_encoding", action="store_true")
    parser.add_argument("--fft", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--after", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser

    
def parser_to_config(parser, config):
    """
    Add parser argument into oemgaconfig
    """
    args = vars(parser.parse_args())
    for key, item in args.items():
        print(key, item)
        config[key] = item
        if key == "lr" or key == "wd":
            config.param_model[key] = item
        if key == "max_epochs":
            config.param_trainer[key] = item
    if config.shots:
        config.param_model.batch_size = config.shots*4
    return config

def main(args):
    # Config File Exist?
    if os.path.exists(args.config_file) == False:
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()
    config = OmegaConf.load(args.config_file) # Create config (similar with args)

    # create directory
    result_dir = os.path.join(args.root_dir, f"{args.result_dirname}/{args.backbone}/{config.exp['data_name']}")
    os.makedirs(result_dir, exist_ok = True)
    result_name = os.path.join(result_dir, f"{args.method}.csv")
    # change args into config
    config = parser_to_config(parser, config) # To preserve config file but input the argument
    #if config.exp.model_type == "spectroresnet":
    #    config = parser_for_spectroresnet(parser, config)
    
    assert not ((config.group_avg) and (config.loss_selection))
    assert not ((config.group_avg) and (config.worst))
    assert not ((config.worst) and (config.loss_selection))
    
    if config.group_avg: # Model Selection Criterion [Default: "val_mse"]
        val_type = "val_group_mse"
        config.objective.type = val_type
        config.param_early_stop.monitor = val_type
        config.logger.param_ckpt.monitor = val_type

    elif config.loss_selection: # Model Selection Criterion [Default: "val_mse"]
        val_type = "train_loss"
        config.objective.type = val_type
        config.param_early_stop.monitor = val_type
        config.logger.param_ckpt.monitor = val_type

    elif config.worst: # Model Selection Criterion [Default: "val_mse"]
        val_type = "val_worst_mae"
        config.objective.type = val_type
        config.param_early_stop.monitor = val_type
        config.logger.param_ckpt.monitor = val_type

    # set seed
    seed_everything(config.seed)
    
    #--- get the solver
    if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
        solver = solver_s2s(config)
    elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp']: # Our Interest
        torch.use_deterministic_algorithms(True)
        solver = solver_s2l(config)
    else:
        solver = solver_f2l(config)

    #--- training and logging into mlflow
    init_mlflow(config) # Ignore mlflow, in experiment for etri, we use csv file in the path of result_name
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics, run_list, logger_list = solver.evaluate() 
        logger.info(cv_metrics)
        mf.log_metrics(cv_metrics)

    # Naming the experiment results which is saved in CSV file.        
        if config.method == 'original':
            cv_metrics['name'] = f"original_lr_{config.lr}_wd_{config.wd}"
        elif config.method == 'prompt_lowgen' or config.method == 'prompt_lowglogen':
            cv_metrics['name'] = f'layer_{config.layer_num}_lr_{config.lr}_wd_{config.wd}'
        elif config.method == 'prompt_global':
            cv_metrics['name'] = f"lr_{config.lr}_wd_{config.wd}"
        elif config.method == 'prompt_gen' or config.method == 'prompt_glogen':
            cv_metrics['name'] = f'lr_{config.lr}_wd_{config.wd}'
        if config.method in ['prompt_gen', 'prompt_glogen', 'prompt_lowglogen', 'prompt_lowgen']:
            cv_metrics['name'] += f'_gencoef_{config.gen_coeff}'
        if config.method in ['prompt_global', 'prompt_glogen', 'prompt_lowglogen' ]:
            cv_metrics['name'] += f'_global_coeff_{config.global_coeff}'

        if config.glonorm:
            cv_metrics['name'] += '_glonorm'
        if config.gennorm:
            cv_metrics['name'] += '_gennorm'
        if config.group_avg:
            cv_metrics['name'] += '_groupavg'
        if config.normalize:
            cv_metrics['name'] += '_norm'
        if config.clip:
            cv_metrics['name'] += '_clip'
        if config.var:
            cv_metrics['name'] += f'_var_{config.var}'
        if config.gvar:
            cv_metrics['name'] += f'_gvar'
        if config.ssvar:
            cv_metrics['name'] += f'_ssvar_{config.ssvar}'
        if config.param_trainer.max_epochs != 100:
            cv_metrics['name'] += f'_epoch_{config.param_trainer.max_epochs}'

        if config.shots:
            cv_metrics['name'] += f'_shot_{config.shots}'
        if config.fft:
            cv_metrics['name'] += f'_fft'
        if config.pca_encoding:
            cv_metrics['name'] += f'_pca_{config.n_components}'
        if config.trigger:
            cv_metrics['name'] += f'_trigger'
        if config.cross:
            cv_metrics['name'] += f'_cross'
        if config.worst:
            cv_metrics['name'] += f'_worst'
        if config.after:
            cv_metrics['name'] += f'_after'
        if config.debug:
            cv_metrics['name'] += f'_debug'
        cv_metrics['name'] += f'_seed_{config.seed}'
    
               
    save_result(cv_metrics, result_name) # Save test result to csv

    # print pretty
    for key in cv_metrics.keys():
        print(f"{key}: {cv_metrics[key]}")
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")

    # clear redundant mlflow models (save disk space)  
    if config.remove:
        for j in run_list:
            if os.path.exists(j):                 
                rmtree(j[:-9])
        
        # # clear redundatn lightning log (save disk space)
        # for j in logger_list:
        #     if os.path.exists(j):
        #         rmtree(j)


if __name__ == '__main__':
    torch.cuda.reset_max_memory_allocated()
    parser = get_parser()
    args = parser.parse_args()
    main(parser.parse_args())
    max_memory_used = max_gpu_memory()
    print(f"Maximum GPU memory used: {max_memory_used / (1024 * 1024):.2f} MB")

