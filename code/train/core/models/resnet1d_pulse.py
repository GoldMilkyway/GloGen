# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base_pl import Regressor
import coloredlogs, logging

import copy

coloredlogs.install()
logger = logging.getLogger(__name__)


class Resnet1d_Pulse(Regressor):
    def __init__(self, param_model, random_state=0):
        super(Resnet1d_Pulse, self).__init__(param_model, random_state)

        self.model = ResNet()
        self.load_output_on = False

    def _shared_step(self, batch):
        x_ppg, y, group, x_abp, peakmask, vlymask = batch
        pred = self.model(x_ppg)
        loss = self.criterion(pred, y)
        if self.load_output_on:
            with torch.no_grad():
                self.embedding = copy.deepcopy(self.model)
                self.embedding.main_clf = nn.Identity()
                embed = self.embedding(x_ppg)
            return loss, pred, x_abp, y, group, x_ppg, embed

        else:
            return loss, pred, x_abp, y, group

    def training_step(self, batch, batch_idx):
        self.step_mode = "train"
        if self.load_output_on:
            loss, pred_bp, t_abp, label, group, x, embed = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss": loss, "pred_bp": pred_bp, "true_abp": t_abp, "true_bp": label,
            "group": group, "x_ppg": x['ppg'], "embed": embed}
        else:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss": loss, "pred_bp": pred_bp, "true_abp": t_abp, "true_bp": label, "group": group}

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

    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        group = torch.cat([v["group"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        self.step_mode = "val"
        if self.load_output_on:
            loss, pred_bp, t_abp, label, group, x, embed = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
            return {"loss": loss, "pred_bp": pred_bp, "true_abp": t_abp, "true_bp": label,
            "group": group, "x_ppg": x['ppg'], "embed": embed}
        else:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss": loss, "pred_bp": pred_bp, "true_abp": t_abp, "true_bp": label, "group": group}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        self.step_mode = "test"
        if self.load_output_on:
            loss, pred_bp, t_abp, label, group, x, embed = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('test_loss', loss, prog_bar=True)
            return {"loss": loss, "pred_bp": pred_bp, "true_abp": t_abp, "true_bp": label,
            "group": group, "x_ppg": x['ppg'], "embed": embed}
        else:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            group = group.unsqueeze(1)
            self.log('test_loss', loss, prog_bar=True)
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
        return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me} 

    def load_all_output(self):
        self.load_output_on = True

    def cancel_all_output(self):
        self.load_output_on = False


def conv3x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """3x1 convolution with padding, output_len=input_len"""
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1, #If dilation =n, then kernel size is equivalent to 3+2n. To keep same output size, use padding=dilation.
        groups=1,
        bias=False,
        dilation=1,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution with no padding, output_len=input_len """
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )
  
class BasicBlock(nn.Module): #BasicBlock always have dilation=1 and groups=1
    expansion: int = 1 #Basic block expect input and output to have same number of channels
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample = None,
        norm_layer = nn.BatchNorm1d,
    ) -> None:
        super(BasicBlock,self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_channels, out_channels, stride=1)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
      
      
class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_BP=2, zero_init_residual=False, norm_layer=nn.BatchNorm1d):
        super(ResNet, self).__init__()

        self._norm_layer = norm_layer
 
        self.input_channels = 64

        
        self.conv1 = nn.Conv1d(1, self.input_channels, kernel_size=7, stride=2, padding=3,
                               bias=False) # 1CH -> 64CH, NowLen->Len/2
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True) 
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) #NowLen->Len/4
        
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=layers[0]) # 64CH->64CH, NowLen=Len/4
        
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=layers[1], stride=2) # 64CH->128CH, NowLen=Len/8
        
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=layers[2], stride=2) # 128CH->256CH, NowLen=Len/16
        
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=layers[3], stride=2) # 256CH->512CH, NowLen=Len/32
        
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Final feature map =512*1
        self.main_clf = nn.Linear(512 * block.expansion, num_BP)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
            
         # Adjust the identity mapping method to match desired number of channels and length   
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        # The first block adapt input channel to output channel, and downsample the length 
        layers.append(block(self.input_channels, out_channels, stride, downsample, norm_layer))
        
        # Next time when _make_layer is called, the input channel is the previous output channel
        self.input_channels = out_channels * block.expansion 
        
        # The rest of blocks do not change length or channels
        for _ in range(1, num_blocks):
            layers.append(block(self.input_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if type(x) == dict:
            x = x['ppg']
        x = x.reshape(len(x), 1, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.main_clf(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def Resnet18_1D(**kwargs):
    return ResNet(**kwargs)