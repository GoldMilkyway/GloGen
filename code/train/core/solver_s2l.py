#%%
import os
import joblib
from shutil import rmtree, copy
import pandas as pd
import numpy as np 
from scipy.io import loadmat
from mat73 import loadmat as loadmat73
from glob import glob
import pytorch_lightning as pl
from sklearn.metrics import r2_score
# Load loaders
from core.loaders import *
from core.solver_s2s import Solver
#####################################################
#####################################################
from core.utils import (get_nested_fold_idx, get_ckpt, cal_metric, cal_statistics, mat2df, mat2df_mimic, to_group,
                        remove_outlier, group_annot, group_count, group_shot, transferring, pulsedb_group_shot)
from core.load_model import model_fold
from core.model_config import model_configuration
from core.prompt_tuning import Custom_model
import pickle
from omegaconf import OmegaConf
#####################################################
#####################################################
# Load model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from core.models import *

# Others
import torch.nn as nn
import torch
import mlflow as mf
import coloredlogs, logging
from pathlib import Path
import warnings
import pdb
from copy import deepcopy
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

# def seed_everything(SEED):
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     pl.utilities.seed.seed_everything(seed=SEED)
#     torch.backends.cudnn.deterministic = True ###
#     torch.backends.cudnn.benchmark = False ###

#%%

    
class SolverS2l(Solver):
    # def __init__(self, config, transfer):
    #     super(SolverS2l, self).__init__()
    #     self.transfer
    def _get_model(self, ckpt_path_abs=None, fold=None):
        model = None
        if self.config.transfer: # Transfer Learning
            if self.config.exp.model_type == "resnet1d":
                backbone_name = f"{self.config.transfer}-{self.config.exp.model_type}"
                if self.config.transfer == "uci2":
                    fold=0
                if self.config.method == "original": # TODO
                    if self.config.scratch:
                        self.transfer_config_path = f"./core/config/dl/resnet/resnet_{self.config.transfer}.yaml"
                        self.transfer_config = OmegaConf.load(self.transfer_config_path)
                        self.transfer_config = transferring(self.config, self.transfer_config)
                        if self.config.exp.data_name in ['mimic', 'vitaldb']:
                            model = Resnet1d_Pulse(self.config.param_model, random_state=self.config.exp.random_state)
                        else:
                            model = Resnet1d_original(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                    else:
                        if self.config.exp.data_name in ['mimic', 'vitaldb']:
                            model = Resnet1d_Pulse.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                        else:
                            model = Resnet1d_original.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                        model.param_model.lr = self.config.param_model.lr
                        model.param_model.wd = self.config.param_model.wd
                        model.param_model.batch_size = self.config.param_model.batch_size
                else:
                    if self.config.exp.data_name in ['mimic', 'vitaldb']:
                        model = Resnet1d_Pulse.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                    else:
                        model = Resnet1d.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                # Initialize Classifier
                model.model.main_clf = nn.Linear(in_features=model.model.main_clf.in_features,
                                                 out_features=model.model.main_clf.out_features,
                                                 bias=model.model.main_clf.bias is not None)
                print(f"####### Load {self.config.exp.model_type} backbone model pre-trained by {self.config.transfer} #######")
            else:
                NotImplementedError
            return model

        elif not ckpt_path_abs: # Pre-training
            if self.config.exp.model_type == "resnet1d":
                if self.config.exp.data_name in ['mimic', 'vitaldb']:
                    model = Resnet1d_Pulse(self.config.param_model, random_state=self.config.exp.random_state)
                else:
                    if self.config.method == "original":
                        model = Resnet1d_original(self.config.param_model, random_state=self.config.exp.random_state)
                    else:
                        model = Resnet1d(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP(self.config.param_model, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else: # IID Fine-tuning
            if self.config.exp.model_type == "resnet1d":
                if self.config.method == "original":
                    model = Resnet1d_original.load_from_checkpoint(ckpt_path_abs)
                else:
                    model = Resnet1d.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP.load_from_checkpoint(ckpt_path_abs)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val"):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm

        #--- Predict
        pred = outputs["pred_bp"].numpy()
        true = outputs["true_bp"].numpy()
        naive =  np.mean(dm.train_dataloader(is_print=False).dataset._target_data, axis=0)
        #####################################################
        #####################################################

        # Make Prediction into Group Prediction
        pred_group_bp, true_group_bp = to_group(pred, true, self.config, loader.dataset.bp_norm)  # TODO
        #####################################################
        #####################################################
        #--- Evaluate
        err_dict = {}
        for i, tar in enumerate(['SP', 'DP']):
            tar_acrny = 'sbp' if tar=='SP' else 'dbp'
            pred_bp = bp_denorm(pred[:,i], self.config, tar)
            true_bp = bp_denorm(true[:,i], self.config, tar)
            naive_bp = bp_denorm(naive[i], self.config, tar)

            # error
            err_dict[tar_acrny] = pred_bp - true_bp
            fold_errors[f"{mode}_{tar_acrny}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar_acrny}_label"].append(true_bp)
            #####################################################
            #####################################################

            # Group Performance
            for group in ["hypo", "normal", "prehyper", "hyper2",]:
                if len(pred_group_bp[tar][group]) != 0:
                    pr_group_bp = bp_denorm(pred_group_bp[tar][group], self.config, tar)
                    tr_group_bp = bp_denorm(true_group_bp[tar][group], self.config, tar)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_pred"].append(pr_group_bp)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_label"].append(tr_group_bp)
            #####################################################
            #####################################################
            fold_errors[f"{mode}_{tar_acrny}_naive"].append([naive_bp]*len(pred_bp))
        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        
        metrics = cal_metric(err_dict, mode=mode)    
        return metrics
            
#%%
    def evaluate(self):
        #####################################################
        #####################################################

        # Make Template considering group sbp and group dbp
        fold_errors_template = {"subject_id":[], "record_id": [],
                                "sbp_naive":[],  "sbp_pred":[], "sbp_label":[],
                                "dbp_naive":[],  "dbp_pred":[],   "dbp_label":[],
                                "sbp_hypo_pred":[], "dbp_hypo_pred":[],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        
        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        elif "mimic" in self.config.exp.subject_dict:
            few_shot_path = "./few_shot_dataset/mimic_few_shot"
            if (not self.config.shots) or (self.config.shots and self.config.create_shot_loader):
                all_split_df = []
                print("###### Loading MIMIC #########")
                for i, k in zip(["Test", "Val", "Train"],["CalFree_Test_Subset", "CalBased_Test_Subset","Train_Subset"]):
                    print(f"$$$$ Loading {i}:{k} $$$$")
                    all_split_df.append(mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}{k}.mat")['Subset']))
                print("###### Loading Finish #########")
                if self.config.create_shot_loader:
                    print("Creating Few shot loader for MIMIC")
                    os.makedirs(few_shot_path, exist_ok=True)
                    test_debug_shot_5, val_shot_5, train_shot_5 = pulsedb_group_shot(all_split_df, 5)
                    test_debug_shot_10, val_shot_10, train_shot_10 = pulsedb_group_shot(all_split_df, 10)
                    # Save into pickle
                    test_debug_shot_5.to_pickle(os.path.join(few_shot_path,"test_debug_shot_5.pkl"))
                    val_shot_5.to_pickle(os.path.join(few_shot_path,"val_shot_5.pkl"))
                    train_shot_5.to_pickle(os.path.join(few_shot_path,"train_shot_5.pkl"))
                    test_debug_shot_10.to_pickle(os.path.join(few_shot_path,"test_debug_shot_10.pkl"))
                    val_shot_10.to_pickle(os.path.join(few_shot_path,"val_shot_10.pkl"))
                    train_shot_10.to_pickle(os.path.join(few_shot_path,"train_shot_10.pkl"))
                    
            if self.config.shots:
                print("### Loading Train Few shot ###")
                train_df = pd.read_pickle(os.path.join(few_shot_path, f"train_shot_{self.config.shots}.pkl"))
                print("### Loading Validation Few shot ###")
                val_df = pd.read_pickle(os.path.join(few_shot_path, f"val_shot_{self.config.shots}.pkl"))
                if self.config.debug:
                    print("### Loading Test Few shot ###")
                    test_df = pd.read_pickle(os.path.join(few_shot_path, f"test_debug_shot_{self.config.shots}.pkl"))
                else:
                    print("### Loading Test Full shot ###")
                    test_df = mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}CalFree_Test_Subset.mat")['Subset'])
                all_split_df = [test_df, val_df, train_df]

        elif "vitaldb" in self.config.exp.subject_dict:
            few_shot_path = "./few_shot_dataset/vitaldb_few_shot"
            if (not self.config.shots) or (self.config.shots and self.config.create_shot_loader):
                all_split_df = []
                print("###### Loading MIMIC #########")
                for i, k in zip(["Test", "Val", "Train"],["CalFree_Test_Subset", "CalBased_Test_Subset","Train_Subset"]):
                    print(f"$$$$ Loading {i}:{k} $$$$")
                    all_split_df.append(mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}VitalDB_{k}.mat")['Subset']))
                print("###### Loading Finish #########")
                if self.config.create_shot_loader:
                    print("Creating Few shot loader for VitalDB")
                    os.makedirs(few_shot_path, exist_ok=True)
                    test_debug_shot_5, val_shot_5, train_shot_5 = pulsedb_group_shot(all_split_df, 5)
                    test_debug_shot_10, val_shot_10, train_shot_10 = pulsedb_group_shot(all_split_df, 10)
                    # Save into pickle
                    test_debug_shot_5.to_pickle(os.path.join(few_shot_path,"test_debug_shot_5.pkl"))
                    val_shot_5.to_pickle(os.path.join(few_shot_path,"val_shot_5.pkl"))
                    train_shot_5.to_pickle(os.path.join(few_shot_path,"train_shot_5.pkl"))
                    test_debug_shot_10.to_pickle(os.path.join(few_shot_path,"test_debug_shot_10.pkl"))
                    val_shot_10.to_pickle(os.path.join(few_shot_path,"val_shot_10.pkl"))
                    train_shot_10.to_pickle(os.path.join(few_shot_path,"train_shot_10.pkl"))
                    
            if self.config.shots:
                print("### Loading Train Few shot ###")
                train_df = pd.read_pickle(os.path.join(few_shot_path, f"train_shot_{self.config.shots}.pkl"))
                print("### Loading Validation Few shot ###")
                val_df = pd.read_pickle(os.path.join(few_shot_path, f"val_shot_{self.config.shots}.pkl"))
                if self.config.debug:
                    print("### Loading Test Few shot ###")
                    test_df = pd.read_pickle(os.path.join(few_shot_path, f"test_debug_shot_{self.config.shots}.pkl"))
                else:
                    print("### Loading Test Full shot ###")
                    test_df = mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}VitalDB_CalFree_Test_Subset.mat")['Subset'])
                all_split_df = [test_df, val_df, train_df]

        #####################################################
        #####################################################
        # Remove outlier
        all_split_df = remove_outlier(all_split_df)
        
        # Make group annotation
        all_split_df = group_annot(all_split_df) 

        if self.config.count_group:
            group_count(all_split_df)

        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df) 
        # print(self.config)


        run_list = []
        logger_list = []

        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            # seed_everything(self.config.seed)
            # get_nested_fold_idx : [[0,1,2],[3],[4]] ## Generator
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            #if foldIdx==1: break
            train_df = pd.concat([all_split_df[i] for i in folds_train])
            val_df = pd.concat([all_split_df[i] for i in folds_val])
            if (not "mimic" in self.config.exp.subject_dict) and (not "vitaldb" in self.config.exp.subject_dict):
                # few_shot dataset for mimic and vitaldb are already created: refer to utils.pulsedb_group_shot
                if self.config.shots or self.config.debug: # train and validate with few-shot
                    if self.config.debug:
                        train_df = group_shot(train_df, n=5)
                    else:
                        train_df = group_shot(train_df, n=self.config.shots)
                    val_df = group_shot(val_df, n=5)
            
            test_df = pd.concat([all_split_df[i] for i in folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)            
            
            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max] 

            #####################################################
            #####################################################
            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                if self.config.transfer:    # Prompt Transfer Learning
                    regressor = self._get_model(fold=foldIdx)
                else:  # TODO               # Prompt Fine-tuning
                    ck_path = os.path.join(self.config.root_dir, "models", model_fold[self.config.backbone][data_name][self.config.seed][foldIdx])
                    regressor = self._get_model(ck_path) # Load Model 

                # Transfer or not
                if self.config.transfer:
                    model_config = model_configuration[self.config.backbone][self.config.transfer] # Load pre-trained model config
                else:
                    model_config = model_configuration[self.config.backbone][data_name] # Load pre-trained model config

                data_shape = model_config["data_dim"] 
                # if self.config.group_avg:
                model = Custom_model(regressor, data_shape, model_config, self.config, stats) # Call Prompt Model 
                # else:
                #     from core.prompt_tuning_temp import Custom_model
                #     model = Custom_model(regressor, data_shape, model_config, self.config, stats) # Call Prompt Model 


                # Gradient Control -- freeze pre-train model // train prompt
                for name, param in model.named_parameters():
                    if 'prompt_learner' not in name:
                        param.requires_grad_(False)
                    if self.config.update_encoder:
                        if 'extractor' in name:
                            param.requires_grad_(True)
                    if self.config.update_regressor:
                        if 'regressor' in name:
                            param.requires_grad_(True)
                    elif self.config.transfer:
                        if 'main_clf' in name:
                            param.requires_grad_(True)
                enabled = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        enabled.add(name)
                print(f"Parameters to be updated: {enabled}")

            if self.config.method == "original":
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:
                    model = self._get_model()
                print("##"*20)
                for name, param in model.named_parameters():
                    
                    # Linear-Probing
                    if self.config.lp: 
                        if not "main_clf" in name:
                            param.requires_grad_(False)
                            send_message = "Linear Probing"

                    # Fine-tuning
                    elif self.config.transfer:
                        send_message = "Fine-tuning"

                    # Scratch
                    else:
                        send_message = "Training from Scratch"
                print(send_message)
                print("##"*20)

                enabled = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        enabled.add(name)
                print(f"Update Param:\n{enabled}")
                print("##"*20)
            #####################################################
            #####################################################
            # Define optimizer with different learning rates for prompt_learner and other parameters

            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop)) # choosing earlystopping policy
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt)) # choosing model save policy
            lr_logger = LearningRateMonitor()

            # Control all training and test process
            
            trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

            #--- trainer main loop
            logger_list.append(trainer.logger.log_dir)
            mf.pytorch.autolog()
            
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # train
                trainer.fit(model, dm) ## Training
                print("run_id", run.info.run_id)
                print("experiment_id", run.info.experiment_id)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id)) 
                # import pdb; pdb.set_trace()
                # ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0]) # mlruns ckpt_path
                # if self.config.shots:
                    # ckpt_path_abs = checkpoint_callback.best_model_path
                ckpt_path_abs = checkpoint_callback.best_model_path
                print("ckpt_path_abs", ckpt_path_abs)


                # ### Sanity Check: Model Weight Update
                # if self.config.sanity_check:
                #     print(f"encoder_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                
                # load best ckpt
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"]) # TODO

                # evaluate
                if self.config.save_outputs:
                    path = f"./save_output/{self.config.transfer}_{self.config.exp.data_name}/"
                    if self.config.method == "prompt_glogen":
                        path = os.path.join(path, "prompt_glogen")
                    elif self.config.lp:
                        path = os.path.join(path, "lp")
                    elif self.config.scratch:
                        path = os.path.join(path, "scratch")
                    elif self.config.method == "original":
                        path = os.path.join(path, "ft")
                    else:
                        NotImplementedError
                    
                    print("Saving Output Path: ", path)

                    os.makedirs(path, exist_ok=True)
                    model.load_all_output()
                    train_outputs = trainer.test(model=model, test_dataloaders=dm.train_dataloader(), verbose=True) ## Test
                    train_outputs_ = deepcopy(train_outputs)
                    model.cancel_all_output()

                    ## denorm ####
                    
                    train_outputs_['x_ppg'] = dm.train_dataloader().dataset.ppg_denorm(train_outputs['x_ppg'], self.config)
                    xx = dm.train_dataloader().dataset.bp_denorm(train_outputs['true_bp'][:,0],self.config,'SP')
                    yy = dm.train_dataloader().dataset.bp_denorm(train_outputs['true_bp'][:,1],self.config,'DP')
                    train_outputs_['true_bp'] = torch.stack((xx,yy),dim=1)
                    
                    ### Save #####
                    with open(f'{path}/train_shot{self.config.shots}_fold{foldIdx}.pkl', 'wb') as f:
                        pickle.dump(train_outputs_, f)

                    val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False) ## Validation

                    model.load_all_output()
                    test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True) ## Test
                    test_outputs_ = deepcopy(test_outputs)
                    model.cancel_all_output()

                    ### denorm ####
                    test_outputs_['x_ppg'] = dm.train_dataloader().dataset.ppg_denorm(test_outputs['x_ppg'], self.config)
                    xx = dm.train_dataloader().dataset.bp_denorm(test_outputs['true_bp'][:,0],self.config,'SP')
                    yy = dm.train_dataloader().dataset.bp_denorm(test_outputs['true_bp'][:,1],self.config,'DP')
                    test_outputs_['true_bp'] = torch.stack((xx,yy),dim=1)

                    with open(f'{path}/test_shot{self.config.shots}_fold{foldIdx}.pkl', 'wb') as f:
                        pickle.dump(test_outputs_, f)
                        
                else:
                    
                    val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False) ## Validation
                    test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True) ## Test


                if self.config.gen_ip: # if gen_ip=True, Save prompt data, ppg_input, global_prompt, instance-wise prompt, merged prompt
                    with open(f"./glogen_glogen_fold_{foldIdx}.pkl", 'wb') as pickle_file:
                        pickle.dump(test_outputs, pickle_file)               
                
                # save updated model
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_abs)

                metrics = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

                redundant_model_path = str(Path(artifact_uri))
                run_list.append(redundant_model_path)
                

            #--- Save to model directory
            if self.config.save_checkpoint:
                os.makedirs("{}-{}/{}".format(self.config.path.model_directory, self.config.method, self.config.exp.exp_name), exist_ok=True)
                ckpt_real = "{}-{}/{}/lr={}_wd={}_fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(self.config.path.model_directory, self.config.method, self.config.exp.exp_name,self.config.lr, self.config.wd, foldIdx,metrics["test/sbp_mae"],metrics["test/dbp_mae"]) 
                trainer.save_checkpoint(ckpt_real)
                print("###"*3, "Save Checkpoint", "###"*3)
                print("Checkpoint Path :", ckpt_real)
                print("###"*8)

        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)

        #####################################################
        #####################################################
        for mode in ['val', 'test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp',  'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            #####################################################
            #####################################################
            tmp_metric = cal_metric(err_dict, mode=mode)
            r2_metric = {f"{mode}_{tar}_r2": r2_score(fold_errors[f"{mode}_{tar}_label"],fold_errors[f"{mode}_{tar}_pred"]) \
                        for tar in ['sbp', 'dbp',  'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            out_metric.update(tmp_metric)
            out_metric.update(r2_metric)

        return out_metric, run_list, logger_list

    def test(self): # Similar with self.evaluate() - all same but no training [trainer.fit()]
        results = {}
        #####################################################
        #####################################################

        fold_errors_template = {"subject_id": [], "record_id": [],
                                "sbp_naive": [], "sbp_pred": [], "sbp_label": [],
                                "dbp_naive": [], "dbp_pred": [], "dbp_label": [],
                                "sbp_hypo_pred": [], "dbp_hypo_pred": [],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]}

        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        elif "mimic" in self.config.exp.subject_dict:
            print("###### Loading MIMIC #########")
            for i, k in zip(["Train", "Val", "Test"],["Train_Subset", "CalBased_Test_Subset", "CalFree_Test_Subset"]):
                print(f"$$$$ Loading {i}:{k} $$$$")
                all_split_df = [mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}{k}.mat"))]
            print("###### Loading Finish #########")
        elif "vitaldb" in self.config.exp.subject_dict:
            print("###### Loading VitalDB #########")
            for i, k in zip(["Train", "Val", "Test"],["Train_Subset", "CalBased_Test_Subset", "CalFree_Test_Subset"]):
                print(f"$$$$ Loading {i}:{k} $$$$")
                all_split_df = [mat2df_mimic(loadmat73(f"{self.config.exp.subject_dict}VitalDB_{k}.mat"))]
            print("###### Loading Finish #########")

        #####################################################
        #####################################################
        all_split_df = remove_outlier(all_split_df)
        all_split_df = group_annot(all_split_df)
        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            #seed_everything(self.config.seed)
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            #if foldIdx == 1 : break
            train_df = pd.concat([all_split_df[i] for i in folds_train])
            val_df = pd.concat([all_split_df[i] for i in folds_val])
            test_df = pd.concat([all_split_df[i] for i in folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max] 

            #--- load trained model
            if 'param_trainer' in self.config.keys():
                trainer = MyTrainer(**dict(self.config.param_trainer))
            else:
                trainer = MyTrainer()

            ckpt_path_abs = glob(f'{self.config.param_test.model_path}{foldIdx}' + '*.ckpt')[0]

            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                if self.config.transfer:
                    regressor = self._get_model(fold=foldIdx)
                else:
                    regressor = self._get_model(ckpt_path_abs) # Load Model # TODO
                model_config = model_configuration[self.config.backbone][data_name] # Load pre-trained model config
                data_shape = model_config["data_dim"] 
                model = Custom_model(regressor, data_shape, model_config, self.config, stats)
                #model = Custom_model.load_from_checkpoint(ckpt_path_abs) # Call Prompt Model 
                #import pdb; pdb.set_trace()
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"])

                #model = model.load_from_checkpoint(ckpt_path_abs)

            if self.config.method == "original":
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:
                    model = self._get_model()

            model.eval()
            trainer.model = model

            #--- get test output
            val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
            test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
            
            metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
            logger.info(f"\t {metrics}")

        #--- compute final metric
        results['fold_errors'] = fold_errors
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        #####################################################
        #####################################################
        for mode in ['test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp', 'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
        #####################################################
        #####################################################
        results['out_metric'] = out_metric
        os.makedirs(os.path.dirname(self.config.param_test.save_path), exist_ok=True)
        joblib.dump(results, self.config.param_test.save_path)

        print(out_metric)

# %%
