import pytorch_lightning as pl
import torch.nn as nn
import torch
import wandb
import copy
from core.utils import perform_pca, project_to_pca_plane

def hook_fn(module, input, output):
    global hidden_output
    hidden_output = output

def normalizer(x, x_prompted):
    x_max = x.max(dim=-1, keepdim=True)[0]; x_min = x.min(dim=-1, keepdim=True)[0] # 256, 1, 1
    x_prompted_max = x_prompted.max(dim=-1, keepdim=True)[0]; x_prompted_min = x_prompted.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    kk = scale*(x_prompted - x_prompted_min) + x_min
    return kk

def global_normalizer(prompt, x_min, x_max):
    x_prompted_max = prompt.max(dim=-1, keepdim=True)[0]; x_prompted_min = prompt.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    norm_prompt = scale*(prompt - x_prompted_min) + x_min
    return norm_prompt

# Global Prompt
class Prompt(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.model_config = model_config

        # Create Global Prompt
        self.prompt = nn.Parameter(torch.randn(1, self.model_config["data_dim"]))

    def forward(self, x):
        bz = x['ppg'].shape[0]
        expanded_tensor = torch.unsqueeze(self.prompt, 0)
        prompt = expanded_tensor.expand(bz, -1, -1) 

        # Normalize Prompt
        if self.config.glonorm:
            prompt = global_normalizer(prompt, self.x_min, self.x_max)

        # Merging
        prompted = x['ppg'] + self.config.global_coeff*prompt
        return prompted

#Prompt Generator
class PromptGEN_Deconv(nn.Module):
    def __init__(self, config, model_config):
        super(PromptGEN_Deconv, self).__init__()

        self.config = config
        self.model_config = model_config

        print(self.model_config)
        print(self.config.backbone)

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(self.model_config["last_dim"], 128, kernel_size=5, stride=3,)
        self.bn1 = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=7,)
        self.bn2 = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4,)
        self.bn3 = nn.BatchNorm1d(32)
        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=5, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)

        return x

class PromptGEN_Deconv_Large(nn.Module):
    def __init__(self, config, model_config):
        super(PromptGEN_Deconv_Large, self).__init__()

        self.config = config
        self.model_config = model_config # {"data_dim": 1250, "last_dim": 512}

        print(self.model_config)
        print(self.config.backbone)

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(self.model_config["last_dim"], 256, kernel_size=4, stride=1,)
        self.bn1 = nn.BatchNorm1d(256)
        self.deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=7, stride=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3,)
        self.bn3 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5, padding=2)
        self.bn4 = nn.BatchNorm1d(32)
        self.deconv5 = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        
        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        x = self.bn5(x)

        return x
    
class PromptGEN_Deconv_Trigger(nn.Module):
    def __init__(self, config, model_config):
        NotImplementedError
        super(PromptGEN_Deconv_Trigger, self).__init__()

        self.config = config
        self.model_config = model_config

        # Trigger
        self.trigger = nn.Parameter(torch.randn(1,128)) # Trigger_dim = 32

        print(self.model_config)
        print(self.config.backbone)
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(self.model_config["last_dim"]+128, 192, kernel_size=5, stride=3,) # trigger vector 128
        self.bn1 = nn.BatchNorm1d(192)
        self.deconv2 = nn.ConvTranspose1d(192, 96, kernel_size=3, stride=7,)
        self.bn2 = nn.BatchNorm1d(96)
        self.deconv3 = nn.ConvTranspose1d(96, 48, kernel_size=5, stride=4,)
        self.bn3 = nn.BatchNorm1d(48)
        self.deconv4 = nn.ConvTranspose1d(48, 1, kernel_size=7, stride=5, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        # Trigger
        expanded_trigger = self.trigger.expand(x.size(0), -1) # (bs, trigger_dim)  
        x = torch.cat((x, expanded_trigger), dim=1)

        # Apply deconvolution layers
        x = x.unsqueeze(-1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        return x

class PromptGEN_Deconv_Trigger_Large(nn.Module):
    def __init__(self, config, model_config):
        NotImplementedError
        super(PromptGEN_Deconv_Trigger_Large, self).__init__()

        self.config = config
        self.model_config = model_config

        # Trigger
        self.trigger = nn.Parameter(torch.randn(1,128)) # Trigger_dim = 128

        print(self.model_config)
        print(self.config.backbone)
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(self.model_config["last_dim"]+128, 256, kernel_size=4, stride=1,)
        self.bn1 = nn.BatchNorm1d(256)
        self.deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=7, stride=7)
        self.bn2 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3,)
        self.bn3 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5, padding=2)
        self.bn4 = nn.BatchNorm1d(32)
        self.deconv5 = nn.ConvTranspose1d(32, 1, kernel_size=7, stride=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        # Trigger
        expanded_trigger = self.trigger.expand(x.size(0), -1) # (bs, trigger_dim)  
        x = torch.cat((x, expanded_trigger), dim=1)

        # Apply deconvolution layers
        x = x.unsqueeze(-1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        x = self.bn5(x)
        return x

class PromptGEN_Deconv_PCA(nn.Module):
    def __init__(self, config):
        super(PromptGEN_Deconv_PCA, self).__init__()

        self.config = config
        #self.model_config = model_config

        #print(self.model_config)
        print(self.config.backbone)

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(config.n_components, 32, kernel_size=5, stride=3,)
        self.bn1 = nn.BatchNorm1d(32)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=7,)
        self.bn2 = nn.BatchNorm1d(16)
        self.deconv3 = nn.ConvTranspose1d(16, 8, kernel_size=5, stride=4,)
        self.bn3 = nn.BatchNorm1d(8)
        self.deconv4 = nn.ConvTranspose1d(8, 1, kernel_size=7, stride=5, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)

        return x

class PromptGEN_Deconv_PCA_Large(nn.Module):
    def __init__(self, config):
        super(PromptGEN_Deconv_PCA_Large, self).__init__()

        self.config = config
        #self.model_config = model_config

        #print(self.model_config)
        print(self.config.backbone)

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(self.config.n_components, 32, kernel_size=4, stride=1,)
        self.bn1 = nn.BatchNorm1d(32)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=7)
        self.bn2 = nn.BatchNorm1d(16)
        self.deconv3 = nn.ConvTranspose1d(16, 8, kernel_size=3, stride=3,)
        self.bn3 = nn.BatchNorm1d(8)
        self.deconv4 = nn.ConvTranspose1d(8, 4, kernel_size=5, stride=5, padding=2)
        self.bn4 = nn.BatchNorm1d(4)
        self.deconv5 = nn.ConvTranspose1d(4, 1, kernel_size=7, stride=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        x = self.bn5(x)

        return x  

class PromptGEN_Deconv_PCA_trigger(nn.Module):
    def __init__(self, config):
        super(PromptGEN_Deconv_PCA_trigger, self).__init__()

        self.config = config
        #self.model_config = model_config

        #print(self.model_config)
        #print(self.config.backbone)

        self.trigger = nn.Parameter(torch.randn(1,config.n_components)) # Trigger_dim = 32

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(config.n_components*2, 64, kernel_size=5, stride=3,)
        self.bn1 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=7,)
        self.bn2 = nn.BatchNorm1d(32)
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=4,)
        self.bn3 = nn.BatchNorm1d(16)
        self.deconv4 = nn.ConvTranspose1d(16, 1, kernel_size=7, stride=5, padding=1)
        self.bn4 = nn.BatchNorm1d(1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Trigger
        expanded_trigger = self.trigger.expand(x.size(0), -1) # (bs, trigger_dim)  
        x = torch.cat((x, expanded_trigger), dim=1)

        # Apply deconvolution layers
        x = x.unsqueeze(-1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        return x
    
class Custom_model(pl.LightningModule):
    def __init__(self, model, data_shape, model_config, config, stats):
        super().__init__()
        self.config = config
        self.regressor = model
        self.data_shape = data_shape
        self.model_config = model_config
        self.load_output_on = False

        if self.config.pca_encoding:
            self.pca_matrix = None
            self.pca_train_mean = 0

        self.ppg_min = stats[0]   # Maximum Amplitude Value of PPG in training set
        self.ppg_max = stats[1]   # Minimum Amplitude Value of PPG in training set
        if self.config.method == "prompt_gen":
            # Encoder Setting
            if not (self.config.pca_encoding or self.config.fft_encoding):
                self.extractor = copy.deepcopy(self.regressor) # Call pre-trained model
                if config.backbone == "resnet1d":
                    self.extractor.model.main_clf = nn.Identity() # Dropping Linear Classifier
                if config.backbone == "mlpbp":
                    self.extractor.model.mlp_head = nn.Identity() # Dropping MLP Classifier
                else:
                    NotImplementedError
            
            # Decoder Setting
            if self.config.exp.data_name in ['mimic', 'vitaldb']:
                if self.config.pca_encoding: 
                    if not self.config.trigger: # GloGen_PCA
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA_Large(self.config)
                    else: # TODO
                        NotImplementedError
                elif self.config.fft_encoding: 
                    NotImplementedError
                else:
                    if not self.config.trigger: # Gen
                        self.prompt_learner_gen = PromptGEN_Deconv_Large(self.config, self.model_config) 
                    else:                       # Gen_Trigger
                        self.prompt_learner_gen = PromptGEN_Deconv_Trigger_Large(self.config, self.model_config)
            else:            
                if self.config.pca_encoding:
                    if not self.config.trigger: # GloGen_PCA
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA(self.config)
                    else: # TODO
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA_trigger(self.config)
                elif self.config.fft_encoding:
                    NotImplementedError
                else:
                    if not self.config.trigger: # GloGen
                        self.prompt_learner_gen = PromptGEN_Deconv(self.config, self.model_config) # Prompt Generator
                    else:                       # GloGen_Trigger
                        self.prompt_learner_gen = PromptGEN_Deconv_Trigger(self.config, self.model_config)

        elif self.config.method == "prompt_global":
            self.prompt_learner_glo = Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max) # Global Prmopt

        elif self.config.method == "prompt_glogen":
            print("GloGen!!")
            # Encoder Setting
            if not (self.config.pca_encoding or self.config.fft_encoding):
                self.extractor = copy.deepcopy(model) # Call pre-trained model
                if config.backbone == "resnet1d":
                    self.extractor.model.main_clf = nn.Identity() # Dropping Linear Classifier
                if config.backbone == "mlpbp":
                    self.extractor.model.mlp_head = nn.Identity() # Dropping MLP Classifier
                else:
                    NotImplementedError
            
            # Decoder Setting
            if self.config.exp.data_name in ['mimic', 'vitaldb']:
                if self.config.pca_encoding: 
                    if not self.config.trigger: # GloGen_PCA
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA_Large(self.config)
                    else: # TODO
                        NotImplementedError
                elif self.config.fft_encoding: 
                    NotImplementedError
                else:
                    if not self.config.trigger: # Gen
                        self.prompt_learner_gen = PromptGEN_Deconv_Large(self.config, self.model_config) 
                    else:                       # Gen_Trigger
                        self.prompt_learner_gen = PromptGEN_Deconv_Trigger_Large(self.config, self.model_config)
            else:            
                if self.config.pca_encoding:
                    if not self.config.trigger: # GloGen_PCA
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA(self.config)
                    else: # TODO
                        self.prompt_learner_gen = PromptGEN_Deconv_PCA_trigger(self.config)
                elif self.config.fft_encoding:
                    NotImplementedError
                else:
                    if not self.config.trigger: # GloGen
                        self.prompt_learner_gen = PromptGEN_Deconv(self.config, self.model_config) # Prompt Generator
                    else:                       # GloGen_Trigger
                        self.prompt_learner_gen = PromptGEN_Deconv_Trigger(self.config, self.model_config)

            self.prompt_learner_glo = Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max) # Global Prompt

        #Loss Function
        self.criterion = nn.MSELoss()

    def hook_fn(self, module, input, output):
        self.hidden_output = output

    def _shared_step(self, batch): # Common process in training, validation and test
        x_ppg, y, group, x_abp, peakmask, vlymask = batch # we also use group annotation
        ppg = x_ppg['ppg']
        if self.config.pca_encoding:
            if self.config.fft:
                ppg = torch.fft.fft(ppg,dim=2)
                ppg = torch.abs(ppg)
            if (self.pca_matrix == None) & (self.step_mode=="val"): # Sanity check stage
                hidden = project_to_pca_plane(ppg, self.sanity_pca_matrix, self.sanity_val_mean)
            else:
                hidden = project_to_pca_plane(ppg, self.pca_matrix, self.pca_train_mean)
            gen_prompt = self.prompt_learner_gen(hidden)   # Instance-wise prompt
            prompted = self.prompt_learner_glo(x_ppg)     # Global prompt
            
            merged = prompted + self.config.gen_coeff*gen_prompt # merging prmopt with original ppg
            if self.config.var: 
                if self.config.gvar and (self.step_mode in ["train", "val"]):
                    ##################################
                    group_set = torch.arange(len(group.unique())).cuda()
                    group_map = (group_set.unsqueeze(1) == group)
                    group_var_bin = []
                    for i in range(len(group_map)):
                        group_var = gen_prompt[group_map[i]].mean(0).squeeze()
                        group_var_bin.append(group_var)
                    group_prompt_var = torch.stack(group_var_bin, dim=0)
                    prompt_var = torch.var(group_prompt_var, dim=0)
                    var_pen = torch.mean(prompt_var)
                    var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                    ##################################
                else:
                    prompt_var = torch.var(gen_prompt, dim=0) # variance penalty
                    ##################################
                    var_pen = torch.mean(prompt_var)
                    var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                    ##################################

        elif self.config.fft_encoding:
            NotImplementedError
        else:
            if self.config.method == "prompt_gen":
                hidden = self.extractor(ppg)      # ppg representation Z
                gen_prompt = self.prompt_learner_gen(hidden)   # Instance-wise prompt
                if self.config.gennorm:
                    gen_prompt = global_normalizer(gen_prompt, self.ppg_min, self.ppg_max)
                merged = ppg + self.config.gen_coeff *  gen_prompt # merging prompt with original ppg
                if self.config.var:
                    if self.config.gvar and (self.step_mode in ["train", "val"]):
                        ##################################
                        group_set = torch.arange(len(group.unique())).cuda()
                        group_map = (group_set.unsqueeze(1) == group)
                        group_var_bin = []
                        for i in range(len(group_map)):
                            group_var = gen_prompt[group_map[i]].mean(0).squeeze()
                            group_var_bin.append(group_var)
                        group_prompt_var = torch.stack(group_var_bin, dim=0)
                        prompt_var = torch.var(group_prompt_var, dim=0)
                        var_pen = torch.mean(prompt_var)
                        var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                    ##################################
                    else:
                        prompt_var = torch.var(gen_prompt, dim=0) # variance penalty
                        ##################################
                        var_pen = torch.mean(prompt_var)
                        var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                        ##################################

            if self.config.method == "prompt_global":
                merged = self.prompt_learner_glo(x_ppg) # Global prompt

            if self.config.method == "prompt_glogen":
                hidden = self.extractor(ppg)       # ppg representation Z
                gen_prompt = self.prompt_learner_gen(hidden)   # Instance-wise prompt
                if self.config.gennorm:
                    gen_prompt = global_normalizer(gen_prompt, self.ppg_min, self.ppg_max)
                prompted = self.prompt_learner_glo(x_ppg)     # Global prompt
                
                merged = prompted + self.config.gen_coeff*gen_prompt # merging prmopt with original ppg
                if self.config.var and (not self.config.after): 
                    if self.config.gvar and (self.step_mode in ["train", "val"]):
                        if self.config.cross: # Don't Use
                            fix_classifier = torch.zeros((gen_prompt.shape[-1], 4))
                            fix_classifier[:, 0] = 1.
                            fix_classifier[:, 1] = 2.
                            fix_classifier[:, 2] = 3.
                            fix_classifier[:, 3] = 4.
                            fix_classifier = fix_classifier.float().cuda()
                            
                            prediction = torch.matmul( gen_prompt.squeeze(), fix_classifier)
                            var_pen = torch.nn.functional.cross_entropy(prediction, group)
                            var_pen = self.config.ssvar*var_pen
                        else:
                            ##################################
                            group_set = group.unique().cuda()
                            group_map = (group_set.unsqueeze(1) == group)
                            group_var_bin = []
                            for i in range(len(group_map)):
                                group_var = torch.relu(gen_prompt[group_map[i]]).mean(0).squeeze()
                                group_var_bin.append(group_var)
                            group_prompt_var = torch.stack(group_var_bin, dim=0)
                            prompt_var = torch.var(group_prompt_var, dim=0)
                            var_pen = torch.mean(prompt_var)
                            print(f"{self.step_mode}: {var_pen.item()}")
                            var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                            ##################################
                    else: # Don't Use
                        prompt_var = torch.var(gen_prompt, dim=0) # variance penalty
                        ##################################
                        var_pen = torch.mean(prompt_var)
                        var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
                        ##################################
        if self.config.normalize:   # normalize merged ppg model -- but our best model did not use this 
            merged = normalizer(ppg, merged)
        if self.config.clip:        # Cut the merged signal by using maximum and minimum ppg values
            merged = torch.clamp(merged, min= self.ppg_min,max=self.ppg_max)

        if self.config.after and self.config.cross: # Don't Use
            ##################################
            group_set = group.unique().cuda()
            group_map = (group_set.unsqueeze(1) == group)
            group_var_bin = []
            for i in range(len(group_map)):
                group_var = torch.relu(merged[group_map[i]]).mean(0).squeeze()
                group_var_bin.append(group_var)
            group_prompt_var = torch.stack(group_var_bin, dim=0)
            prompt_var = torch.var(group_prompt_var, dim=0)
            var_pen = torch.mean(prompt_var)
            print(f"{self.step_mode}: {var_pen.item()}")
            var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
            ##################################

        elif self.config.after and self.config.gvar: # Don't Use
            ##################################
            group_set = group.unique().cuda()
            group_map = (group_set.unsqueeze(1) == group)
            group_var_bin = []
            for i in range(len(group_map)):
                group_var = torch.relu(merged[group_map[i]]).mean(0).squeeze()
                group_var_bin.append(group_var)
            group_prompt_var = torch.stack(group_var_bin, dim=0)
            prompt_var = torch.var(group_prompt_var, dim=0)
            var_pen = torch.mean(prompt_var)
            print(f"{self.step_mode}: {var_pen.item()}")
            var_pen = self.config.ssvar*torch.nn.functional.relu(self.config.var - var_pen)
            ##################################

        pred = self.regressor(merged)

        loss = self.criterion(pred, y)
        #####################################
        if self.config.var: # variance penalty
            loss += var_pen
        #####################################
        if self.config.gen_ip or self.load_output_on:
            with torch.no_grad():
                self.embedding = copy.deepcopy(self.regressor)
                self.embedding.model.main_clf = nn.Identity()
                embed_merged = self.embedding(merged)
                embed = self.embedding(ppg)
            return loss, pred, x_abp, y, group, ppg, prompted, gen_prompt, merged, embed, embed_merged
        else:
            return loss, pred, x_abp, y, group
        ######################

        
    def grouping(self, losses, group):
        # Make losses into group losses
        group = group.squeeze()
        group_type = torch.arange(0,4).cuda()
        group_map = (group_type.view(-1,1)==group).float()
        group_count = group_map.sum(1)
        group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
        group_loss = group_loss_map.sum(1)                          # (4,2)

        # Average only across the existing group
        mask = group_count != 0
        avg_per_group = torch.zeros_like(group_loss)
        avg_per_group[mask, :] = group_loss[mask, :] / group_count[mask].unsqueeze(1)
        exist_group = mask.sum()
        avg_group = avg_per_group.sum(0)/exist_group
        loss = avg_group.sum()/2
        return loss
    
    def worst_grouping(self, losses, group):
        # Make losses into group losses
        group = group.squeeze()
        group_type = torch.arange(0,4).cuda()
        group_map = (group_type.view(-1,1)==group).float()
        group_count = group_map.sum(1)
        group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
        group_loss = group_loss_map.sum(1)                          # (4,2)
        loss_per_group = group_loss.sum(1)/group_count
        worst_loss = loss_per_group.max()
        return worst_loss
        
    def training_step(self, batch, batch_idx):
        self.step_mode = "train"
        if self.config.pca_encoding:
            if (self.pca_matrix==None):
                assert len(batch[0]['ppg']==self.config.param_model.batch_size)
                self.pca_matrix, self.pca_train_mean = perform_pca(batch[0]['ppg'], n_components=64)
        # if self.config.group_avg:
        #     loss, pred_bp, t_abp, label, group = self._shared_step(batch)
        #     group = group.unsqueeze(1)
        #     self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        #     return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        # else:
        if self.config.gen_ip or self.load_output_on: # Save prompts information  (for quantitative analysis)
            loss, pred_bp, t_abp, label, group, x, gp, ip, merged, embed, embed_merged = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            #wandb.log({'train_loss': loss})
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, 
                    "x_ppg": x, "gp": gp, "ip":ip, "merged": merged, "embed": embed, "embed_merged": embed_merged}    
        else: # Don't save prompts information 
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            group = group.unsqueeze(1)
            #wandb.log({'train_loss': loss})
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}    
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        group = torch.cat([v["group"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        self.step_mode = "val"
        if self.config.pca_encoding:
            if (self.pca_matrix == None):
                self.sanity_pca_matrix = torch.randn((batch[0]['ppg'].shape[-1], 64)).cuda()
                self.sanity_val_mean = torch.mean(batch[0]['ppg'], dim=0)
            
        # if self.config.group_avg:
        #     loss, pred_bp, t_abp, label, group = self._shared_step(batch)
        #     group = group.unsqueeze(1)
        #     self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        #     return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        # else:
        if self.config.gen_ip or self.load_output_on :
            loss, pred_bp, t_abp, label, group, x, gp, ip, merged, embed, embed_merged = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            #wandb.log({'train_loss': loss})
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, 
                    "x_ppg": x, "gp": gp, "ip":ip, "merged": merged, "embed": embed, "embed_merged": embed_merged}           
        else:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
            #wandb.log({'train_loss': loss})
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group} 

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        self.step_mode = "test"
        # if self.config.group_avg:
        #     loss, pred_bp, t_abp, label, group = self._shared_step(batch)
        #     group = group.unsqueeze(1)
        #     self.log('test_loss', loss, prog_bar=True)
        #     return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        # else:
        if self.config.gen_ip or self.load_output_on:
            loss, pred_bp, t_abp, label, group, x, gp, ip, merged, embed, embed_merged = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "x_ppg": x, "gp": gp, "ip":ip, 
            "merged": merged, "embed": embed, "embed_merged": embed_merged}           
        else:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group} 

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="test")
        return test_step_end_out
    
    def _cal_metric(self, logit: torch.tensor, label: torch.tensor, group=None):
        prev_mse = (logit-label)**2
        prev_mae = torch.abs(logit-label)
        prev_me = logit-label
        mse = torch.mean(prev_mse)
        mae = torch.mean(prev_mae)
        me = torch.mean(prev_me)
        std = torch.std(torch.mean(logit-label, dim=1))
        group_mse = self.grouping(prev_mse, group)
        group_mae = self.grouping(prev_mae, group)
        group_me = self.grouping(prev_me, group)
        worst_group_mse = self.worst_grouping(prev_mse, group)
        worst_group_mae = self.worst_grouping(prev_mae, group)
        return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me, "worst_mse":worst_group_mse, "worst_mae":worst_group_mae} 
    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def load_all_output(self):
        self.load_output_on = True

    def cancel_all_output(self):
        self.load_output_on = False

    def configure_optimizers(self): # We use Adam
        if self.config.method == "prompt_global":
            optimizer = torch.optim.Adam([
                {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        if (self.config.method == "prompt_gen"):
            optimizer = torch.optim.Adam([
                {'params': self.prompt_learner_gen.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        if (self.config.method == "prompt_glogen"):
            update_list = [ {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},
                                {'params': self.prompt_learner_gen.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd}]
            if self.config.update_encoder:
                update_list.append({'params': self.extractor.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd})
            elif self.config.update_regressor:
                update_list.append({'params': self.regressor.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd})
            if (self.config.transfer != None) & (not self.config.update_regressor):
                update_list.append({'params': self.regressor.model.main_clf.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd})
            optimizer = torch.optim.Adam(update_list)
        return optimizer
