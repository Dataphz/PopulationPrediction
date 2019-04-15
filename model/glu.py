import torch
from torch import nn 

import random
import numpy as np 

from model.temporally_models import GLU
from model.graph_models import MessageAttentionPassing
from core.config import PreliminaryConfig as cfg 

class MediumBiGLUModel(nn.Module):
    """
    前21天和下一个星期开始的后21天作为输入预测当前
    """
    def __init__(self, c_in=3, fw_input_days = cfg.DATASET.T, bw_input_days = cfg.DATASET.T):
        super(MediumBiGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.FT = fw_input_days
        self.BT = bw_input_days
        self.nodes = 1
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        #Forward
        # block 1
        self.FWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.FWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.FWLN1 = nn.LayerNorm([64, self.nodes, self.FT-2*(self.ks1-1)])
        # block 2
        self.FWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.FWLN2 = nn.LayerNorm([128, self.nodes, self.FT-2*(self.ks1-1) - 1*(self.ks2-1)])
        
        #Barkward
        # block1
        self.BWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.BWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.BWLN1 = nn.LayerNorm([64, self.nodes, self.BT-2*(self.ks1-1)])

        # block 2
        self.BWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.BWLN2 = nn.LayerNorm([128, self.nodes, self.BT-2*(self.ks1-1) - 1*(self.ks2-1)]) # bs x 128 x 1 x (21-3*6)

        # output 
        self.fuseT = self.FT + self.BT - 2 * (2*(self.ks1-1) + 1*(self.ks2-1))

        self.predict1 = nn.Linear(128*self.fuseT, 128)
        self.predict2 = nn.Linear(128, self.predict_days*3)

    
    def forward(self, forward_x, backward_x):
        # forward exact
        fx = self.FWGLU11(forward_x)
        fx = self.FWGLU12(fx)
        fx = self.FWLN1(fx)
        fx = self.FWGLU22(fx)
        fx = self.FWLN2(fx)

        # backward exact
        bx = self.BWGLU11(backward_x)
        bx = self.BWGLU12(bx)
        bx = self.BWLN1(bx)
        bx = self.BWGLU22(bx)
        bx = self.BWLN2(bx)
        
        fuse_feature = torch.cat((fx, bx), dim=3)
        # bs x H x N x T --> bs x N x (T*H)
        bs = fuse_feature.size(0)
        fuse_feature = fuse_feature.permute(0,2,3,1).reshape(bs, -1)# self.nodes,
        hidden_feature = self.predict1(fuse_feature)
        y = self.predict2(hidden_feature)
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class ShortBiGLUModel(nn.Module):
    """
    前14天和下一个星期开始的后14天作为输入预测当前
    """
    def __init__(self, c_in=3, fw_input_days = cfg.DATASET.T, bw_input_days = cfg.DATASET.T):
        super(ShortBiGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.FT = fw_input_days
        self.BT = bw_input_days
        self.nodes = 1
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        #Forward
        # block 1
        self.FWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.FWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.FWLN1 = nn.LayerNorm([64, self.nodes, self.FT-2*(self.ks1-1)])
        # block 2
        # self.FWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        # self.FWLN2 = nn.LayerNorm([128, self.nodes, self.FT-2*(self.ks1-1) - 1*(self.ks2-1)])
        
        #Barkward
        # block1
        self.BWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.BWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.BWLN1 = nn.LayerNorm([64, self.nodes, self.BT-2*(self.ks1-1)])

        # block 2
        # self.BWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        # self.BWLN2 = nn.LayerNorm([128, self.nodes, self.BT-2*(self.ks1-1) - 1*(self.ks2-1)]) # bs x 128 x 1 x (21-3*6)


        # output 
        self.fuseT = self.FT + self.BT - 2 * (2*(self.ks1-1) + 0*(self.ks2-1))

        # self.predict1 = nn.Linear(128*self.fuseT, 128)
        # self.predict2 = nn.Linear(128, self.predict_days*3)

        self.predict1 = nn.Linear(64*self.fuseT, 64)
        self.predict2 = nn.Linear(64, self.predict_days*3)

    
    def forward(self, forward_x, backward_x):
        # forward exact
        fx = self.FWGLU11(forward_x)
        fx = self.FWGLU12(fx)
        fx = self.FWLN1(fx)
        # fx = self.FWGLU22(fx)
        # fx = self.FWLN2(fx)

        # backward exact
        bx = self.BWGLU11(backward_x)
        bx = self.BWGLU12(bx)
        bx = self.BWLN1(bx)
        # bx = self.BWGLU22(bx)
        # bx = self.BWLN2(bx)
        
        fuse_feature = torch.cat((fx, bx), dim=3)
        # bs x H x N x T --> bs x N x (T*H)
        bs = fuse_feature.size(0)
        fuse_feature = fuse_feature.permute(0,2,3,1).reshape(bs, -1)# self.nodes,
        hidden_feature = self.predict1(fuse_feature)
        y = self.predict2(hidden_feature)
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class BiGLUModel(nn.Module):
    """
    30天作为输入
    """
    def __init__(self, c_in=3, fw_input_days = cfg.DATASET.T, bw_input_days = cfg.DATASET.T):
        super(BiGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.FT = fw_input_days
        self.BT = bw_input_days
        self.nodes = 1
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        # Forward
        # block 1
        self.FWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.FWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.FWLN1 = nn.LayerNorm([64, self.nodes, self.FT-2*(self.ks1-1)])
        # block 2
        self.FWGLU21 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        self.FWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.FWLN2 = nn.LayerNorm([128, self.nodes, self.FT-2*(self.ks1-1) - 2*(self.ks2-1)])
        self.FWGLU31 = GLU(c_in=128, c_out=128, ks=4, pd=0)

        # Backward
        # block 1
        self.BWGLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.BWGLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.BWLN1 = nn.LayerNorm([64, self.nodes, self.BT-2*(self.ks1-1)])
        # block 2
        self.BWGLU21 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        self.BWGLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.BWLN2 = nn.LayerNorm([128, self.nodes, self.BT-2*(self.ks1-1) - 2*(self.ks2-1)])
        self.BWGLU31 = GLU(c_in=128, c_out=128, ks=4, pd=0)

        # output 
        self.fuseT = self.FT + self.BT - 2 * (2*(self.ks1-1) + 2*(self.ks2-1)+ 3)
        self.predict1 = nn.Linear(128*self.fuseT, 128)
        self.predict2 = nn.Linear(128, self.predict_days*3)


    def forward(self, forward_x, backward_x):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """

        # forward exact
        fx = self.FWGLU11(forward_x)
        fx = self.FWGLU12(fx)
        fx = self.FWLN1(fx)
        fx = self.FWGLU21(fx)
        fx = self.FWGLU22(fx)
        fx = self.FWLN2(fx)
        fx = self.FWGLU31(fx)


        # backward exact
        bx = self.BWGLU11(backward_x)
        bx = self.BWGLU12(bx)
        bx = self.BWLN1(bx)
        bx = self.BWGLU21(bx)
        bx = self.BWGLU22(bx)
        bx = self.BWLN2(bx)
        bx = self.BWGLU31(bx)

        fuse_feature = torch.cat((fx, bx), dim=3)
        # bs x H x N x T --> bs x N x (T*H)
        bs = fuse_feature.size(0)
        fuse_feature = fuse_feature.permute(0,2,3,1).reshape(bs, -1)
        hidden_feature = self.predict1(fuse_feature) 
        y = self.predict2(hidden_feature)
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class PeriodGLUModel(nn.Module):
    """
    采样点作为输入
    隔7天采样(30,60)--->(4, 8)
    """
    def __init__(self, c_in=3, input_days = cfg.DATASET.T):
        super(PeriodGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days // 7
        self.nodes = cfg.DATASET.nodes 
        self.predict_days = cfg.DATASET.predict_days
        
        self.ks1 = 4
        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        # self.GLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([32, self.nodes, self.T-1*(self.ks1-1)])

        # output 
        self.predict = nn.Linear(32*(self.T-1*(self.ks1-1)), self.predict_days*3)
        
    def forward(self, _input, attention, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        np_x = _input.cpu().data.numpy()
        np_week_x = np_x[:,:,:,2::7]
        # np_week_x = np_x[:,:,:,4::7]
        week_x = torch.FloatTensor(np_week_x).cuda()

        # block 1
        x = self.GLU11(week_x)
        # x = self.GLU12(x)        
        x = self.LN1(x)

        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, self.nodes, -1)
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class ShortGLUModel(nn.Module):
    """
    14天作为输入
    """
    def __init__(self, c_in=3, input_days=30):
        super(ShortGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days
        self.nodes = cfg.DATASET.nodes
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.GLU21 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1)])

        # output 
        self.predict = nn.Linear(64*(self.T-2*(self.ks1-1)), self.predict_days*3)

    def forward(self, x, attention, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        # block 1
        x = self.GLU11(x)
        x = self.GLU21(x)
        x = self.LN1(x)

        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, self.nodes, -1)
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class MediumGLUModel(nn.Module):
    """
    21天作为输入
    """
    def __init__(self, c_in=3, input_days=30):
        super(MediumGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days
        self.nodes = 1#cfg.DATASET.nodes
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7#14
        self.ks2 = 7#4

        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        self.GLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1)])

        # block 2
        # self.GLU21 = GLU(c_in=32, c_out=64, ks=self.ks2, pd=0)
        self.GLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.LN2 = nn.LayerNorm([128, self.nodes, self.T-2*(self.ks1-1) - 1*(self.ks2-1)])

        # output 
        self.predict = nn.Linear(128*(self.T-2*(self.ks1-1) - 1*(self.ks2-1)), self.predict_days*3)

    def forward(self, x, attention, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        # block 1
        x = self.GLU11(x)
        x = self.GLU12(x)
        x = self.LN1(x)

        # block 2 
        # x = self.GLU21(x)
        x = self.GLU22(x)
        x = self.LN2(x)
        
        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, -1)# self.nodes,
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

class GLUModel(nn.Module):
    """
    30天作为输入
    """
    def __init__(self, c_in=3, input_days=30, nodes = cfg.DATASET.nodes):
        super(GLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days
        self.nodes = 1#nodes
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        # self.MAP1 = MessageAttentionPassing(c_in=16, c_out=32)
        self.GLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        # self.GLU13 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1)])

        # block 2
        self.GLU21 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        # self.MAP2 = MessageAttentionPassing(c_in=32, c_out=32)
        self.GLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.LN2 = nn.LayerNorm([128, self.nodes, self.T-2*(self.ks1-1) - 2*(self.ks2-1)])

        # output 
        self.GLU31 = GLU(c_in=128, c_out=128, ks=4, pd=0)#6
        self.predict = nn.Linear(128*(self.T-2*(self.ks1-1) - 2*(self.ks2-1)-3), self.predict_days*3)
        # self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=[0,0])
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, attention=None, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        # block 1
        x = self.GLU11(x)
        # x = self.MAP1(x, attention, constant_attention)
        x = self.GLU12(x)
        x = self.LN1(x)

        # block 2 
        x = self.GLU21(x)
        # x = self.MAP2(x, attention, constant_attention)
        x = self.GLU22(x)
        x = self.LN2(x)

        # output
        x = self.GLU31(x)
        
        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, -1)#self.nodes, -1)
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])#
        return y

class LSGLUModel(nn.Module):
    """
    30天作为输入
    """
    def __init__(self, c_in=3, input_days=30, nodes = cfg.DATASET.nodes):
        super(LSGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days
        self.nodes = 1# nodes
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 14  
        self.ks2 = 4

        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)
        # self.MAP1 = MessageAttentionPassing(c_in=16, c_out=32)
        self.GLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        # self.GLU13 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1)])

        # block 2
        # self.GLU21 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        # self.MAP2 = MessageAttentionPassing(c_in=32, c_out=32)
        self.GLU22 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.LN2 = nn.LayerNorm([128, self.nodes, self.T-2*(self.ks1-1) - 1*(self.ks2-1)])

        # output 
        # self.GLU31 = GLU(c_in=128, c_out=128, ks=6, pd=0)
        self.predict = nn.Linear(128*(self.T-2*(self.ks1-1) - 1*(self.ks2-1)), self.predict_days*3)

    def forward(self, x, attention=None, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        # block 1
        x = self.GLU11(x)
        # x = self.MAP1(x, attention, constant_attention)
        x = self.GLU12(x)
        x = self.LN1(x)

        # block 2 
        # x = self.GLU21(x)
        # x = self.MAP2(x, attention, constant_attention)
        x = self.GLU22(x)
        x = self.LN2(x)

        # output
        # x = self.GLU31(x)
        
        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, -1)# self.nodes,
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y


class LongGLUModel(nn.Module):
    """
    49天作为输入
    """
    def __init__(self, c_in=3, input_days=30):
        super(LongGLUModel, self).__init__()
        seed = cfg.TRAIN.seed#87
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.T = input_days
        self.nodes = cfg.DATASET.nodes
        self.predict_days = cfg.DATASET.predict_days
        self.ks1 = 7
        self.ks2 = 7

        # block 1
        self.GLU11 = GLU(c_in=c_in, c_out=32, ks=self.ks1, pd=0)

        self.GLU12 = GLU(c_in=32, c_out=64, ks=self.ks1, pd=0)
        self.LN1 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1)])

        # block 2
        self.GLU21 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        self.GLU22 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        self.LN2 = nn.LayerNorm([64, self.nodes, self.T-2*(self.ks1-1) - 2*(self.ks2-1)])

        # block 3
        self.GLU31 = GLU(c_in=64, c_out=64, ks=self.ks2, pd=0)
        self.GLU32 = GLU(c_in=64, c_out=128, ks=self.ks2, pd=0)
        self.LN3 = nn.LayerNorm([128, self.nodes, self.T- 4*(self.ks1-1) - 2*(self.ks2-1)])

        # output 
        self.GLU41 = GLU(c_in=128, c_out=128, ks=7, pd=0)
        self.GLU42 = GLU(c_in=128, c_out=128, ks=7, pd=0)
        self.LN4 = nn.LayerNorm([128, self.nodes, self.T- 4*(self.ks1-1) - 2*(self.ks2-1)-12])
        self.predict = nn.Linear(128*(self.T-4*(self.ks1-1) - 2*(self.ks2-1) - 12), self.predict_days*3)

    def forward(self, x, attention, constant_attention=True):
        """
        Params:
            x: [bs x 3 x N x T]
            attention: [bs x 1 x N x N x T]
        Output:
            y: [bs x 3 x N x 15]
        """
        # block 1
        x = self.GLU11(x)
        # x = self.MAP1(x, attention, constant_attention)
        x = self.GLU12(x)
        x = self.LN1(x)

        # block 2 
        x = self.GLU21(x)
        x = self.GLU22(x)
        x = self.LN2(x)

        # block 3
        x = self.GLU31(x)
        x = self.GLU32(x)
        x = self.LN3(x)

        # output
        # x = self.GLU41(x)
        # x = self.GLU42(x)
        # x = self.LN4(x)
        
        # bs x H x N x T --> bs x N x (T*H)
        bs = x.size(0)
        x = x.permute(0,2,3,1).reshape(bs, self.nodes, -1)
        y = self.predict(x) # bs x N x 15*3
        y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
        return y

# class DenseModule(nn.Module):
#     def __init__(self, c_in, c_out):
        
# class googleModule(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(googleModule, self).__init__()
#         self.hidden = c_out // 2

#         # small kernel 6
#         # self.GLU11 = GLU(c_in=c_in, c_out=self.hidden, ks=3, pd=0)
#         # self.GLU12 = GLU(c_in=self.hidden, c_out=self.hidden, ks=3, pd=0)
#         # self.GLU13 = GLU(c_in=self.hidden, c_out=self.hidden, ks=3, pd=0)

#         # medium kernel
#         self.GLU21 = GLU(c_in=c_in, c_out=self.hidden, ks=7, pd=0)
#         self.GLU22 = GLU(c_in=self.hidden, c_out=self.hidden, ks=7, pd=0)
#         # self.GLU23 = GLU(c_in=self.hidden, c_out=self.hidden, ks=2, pd=0)

#         # big
#         self.GLU31 = GLU(c_in=c_in, c_out=self.hidden, ks=13, pd=0)

#     def forward(self, input):
#         # x1 = self.GLU11(input)
#         # x1 = self.GLU12(x1)
#         # x1 = self.GLU13(x1)

#         x2 = self.GLU21(input)
#         x2 = self.GLU22(x2)

#         x3 = self.GLU31(input)

#         feature = torch.cat((x2,x3), dim=1)
#         return feature

# class googleGLUModel(nn.Module):
#     """
#         不行，会出现普遍下降，周期信号会出现上升。
#     """
#     def __init__(self, c_in=3, input_days=cfg.DATASET.T):
#         super(googleGLUModel, self).__init__()
#         self.T = input_days
#         self.nodes = cfg.DATASET.nodes
#         self.predict_days = cfg.DATASET.predict_days
#         self.ks1 = 3
#         self.ks2 = 5

#         # block 1
#         self.module1 = googleModule(c_in=c_in, c_out=64)
#         self.LN1 = nn.LayerNorm([64, self.nodes, self.T-12])

#         # block 2
#         # self.module2 = googleModule(c_in=16, c_out=32)
#         # self.LN2 = nn.LayerNorm([32, self.nodes, self.T-6*2])

#         # block 3
#         # self.module3 = googleModule(c_in=32, c_out=64)
#         # self.LN3 = nn.LayerNorm([64, self.nodes, self.T-6*3])

#         # block 4
#         self.module4 = googleModule(c_in=64, c_out=128)
#         self.LN4 = nn.LayerNorm([128, self.nodes, self.T-12*2])

#         # output 
#         # self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 7), padding=[0,0])
#         # self.sigmoid = nn.Sigmoid()
#         self.predict = nn.Linear(128*(self.T-12*2), self.predict_days*3)

#     def forward(self, x, attention, constant_attention=True):
#         """
#         Params:
#             x: [bs x 3 x N x T]
#             attention: [bs x 1 x N x N x T]
#         Output:
#             y: [bs x 3 x N x 15]
#         """
#         # block 1
#         x = self.module1(x)
#         x = self.LN1(x)

#         # block 2 
#         # x = self.module2(x)
#         # x = self.LN2(x)

#         # block 3
#         # x = self.module3(x)
#         # x = self.LN3(x)

#         # block 3
#         x = self.module4(x)
#         x = self.LN4(x)

#         # output
#         # x = self.sigmoid(self.conv1(x))
        
#         # bs x H x N x T --> bs x N x (T*H)
#         bs = x.size(0)
#         x = x.permute(0,2,3,1).reshape(bs, self.nodes, -1)
#         y = self.predict(x) # bs x N x 15*3
#         y = y.reshape(bs, self.nodes, self.predict_days, 3).permute([0,3,1,2])
#         return y