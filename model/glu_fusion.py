import torch 
from torch import nn 

import random 
import numpy as np 

from core.config import PreliminaryConfig as cfg 
from model.glu import GLUModel, PeriodGLUModel
from model.temporally_models import GLU


# class trendAttentionModule(nn.Module):
#     def __init__(self, c_in=3):
#         self.total_days = cfg.DATASET.T
#         self.GLU = GLU(c_in=c_in, input_days=self.total_days)
    
#     def forward(self, x):
#         """
#             x: [ bs x 1 ]
#         """

class GLUFusionModel(nn.Module):
    def __init__(self, c_in=3):
        super(GLUFusionModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.total_days = cfg.DATASET.T
        # self.recent_days = 7
        # self.week_days =  cfg.DATASET.T // 7

        self.nodes = cfg.DATASET.nodes

        self.tDGLU = GLUModel(c_in=c_in, input_days=self.total_days).cuda()
        self.wDGLU = PeriodGLUModel(c_in=c_in, input_days=self.total_days).cuda()
        # GLU(c_in=c_in, c_out=3, ks=4, pd=0)

        self.tW = nn.Parameter(torch.Tensor(1, 3, self.nodes))
        # self.wW = nn.Parameter(torch.Tensor(1, 3, self.nodes))
        self.init_params()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_params(self):
        self.tW.data.uniform_(-1, 1)
        # self.wW.data.uniform_(-1, 1)
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def forward(self, x, attention, constant_attention=True):
        """
        Params:
            x: [ bs x 3 x N x T]
            attention: did not use now
        """

        tlogits = self.tDGLU(x, attention, constant_attention=constant_attention)
        # rlogits = self.rDGLU(recent_x)#bs x 3 x N x 1
        wlogits = self.wDGLU(x, attention, constant_attention=constant_attention)
        # print(self.tW, self.rW, self.wW)
        # preds = tlogits[:,:,:,0] * self.sigmoid(self.tW) + wlogits[:,:,:,0] * (1.0-self.sigmoid(self.tW))
        preds = tlogits[:,:,:,0]  + wlogits[:,:,:,0] * self.tanh(self.tW)

        return preds.unsqueeze(-1)