import torch
from torch import nn
import torch.nn.functional as F 
from torch.autograd import Variable

from core.config import PreliminaryConfig as cfg 
class MessageAttentionPassing(nn.Module):
    def __init__(self, c_in, c_out):
        super(MessageAttentionPassing, self).__init__()
        
        self.hidden_num = 64
        self.nodes = cfg.DATASET.nodes

        self.atten_fc = nn.Linear(self.nodes, self.nodes)
        self.agg_fc1 = nn.Linear(c_in, self.hidden_num)
        self.agg_fc2 = nn.Linear(self.hidden_num*2, c_out)
        
        self.ReLU = nn.ReLU()

    def forward(self, x, attention, constant_attention=True):
        """
        Params:
            x: [bs x H x N x T ]
            attention: [bs x 1 x N x N x T]
        Output:
            ret: [bs x H x N x T]
        """

        # attention function
        # if not constant_attention:
        #     attention = attention # bs x T x N x N x 1
        
        x = x.permute([0,2,3,1]) #bs x T x N x H
        attention = attention.permute([0,2,3,4,1]) #bs x N x N x T x 1
        
        # message function
        hidden_feature = self.agg_fc1(x) #bs x N x T x H1
        h_feature = hidden_feature.unsqueeze(1).repeat(1,self.nodes,1,1,1) #bs x 1 x N x T x H1-->repeat bs x N x N x T x H1
        v_feature = hidden_feature.unsqueeze(2).repeat(1,1,self.nodes,1,1) #bs x N x 1 x T x H1-->repeat bs x N x N x T x H1
        concat_feature = torch.cat((v_feature,h_feature), dim=-1) #bs x N x N x T x 2*H1
        message_feature = self.agg_fc2(concat_feature) # bs x N x N x T x c_out

        update_feature = torch.sum(attention * message_feature, dim=2) # bs x N x N x T x c_out
        update_nodes = update_feature.permute([0,3,1,2])

        return update_nodes

