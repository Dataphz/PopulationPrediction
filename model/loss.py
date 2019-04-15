import numpy as np 
import torch 
from torch import nn

from core.config import PreliminaryConfig as cfg

def MAPE(v, v_):
    return np.mean(np.abs(v_ - v) / (v + 1e-5))

def MSE(v, v_):
    return np.sqrt(np.mean((v_ - v)**2))

def MAE(v, v_):
    return np.mean(np.abs(v_ - v))

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, preds, labels):
        if cfg.DATASET.flow_transform:
            return torch.sqrt(torch.mean( (preds - labels)**2 ))
        else:
            return torch.sqrt(torch.mean( (torch.log(1+preds) - torch.log(1+labels))**2 ))

class DiffRMSLELoss(nn.Module):
    def __init__(self):
        super(DiffRMSLELoss, self).__init__()

    def forward(self, diff_preds, base, labels):
        """
        Params:
            diff_preds: [ bs x 3 x N x T]
            base: [bs x 3 x N x T]
            labels: [bs x 3 x N x T]
        """

        preds = base + diff_preds
        return torch.sqrt(torch.mean( (preds - labels)**2 ))
        # return torch.sqrt(torch.mean((torch.log(1+preds) - torch.log(1+labels))**2 ))

class DiffRMSLELossVAL(nn.Module):
    def __init__(self):
        super(DiffRMSLELossVAL, self).__init__()

    def forward(self, diff_preds, base, labels):
        """
        Params:
            diff_preds: [ bs x 3 x N x T]
            base: [bs x 3 x N x T]
            labels: [bs x 3 x N x T]
        """

        preds = base + diff_preds
        return torch.sqrt(torch.mean( (preds - labels)**2 ))
        # return torch.sqrt(torch.mean((torch.log(1+preds) - torch.log(1+labels))**2 ))

    