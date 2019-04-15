import torch
from torch import nn

class GLU(nn.Module):
    """
    Wait to add:
        ReLU
    """
    def __init__(self, c_in, c_out, ks=3, pd=1):
        super(GLU, self).__init__()
        self.c_out = c_out
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out*2, kernel_size=(1, ks), padding=(0, pd))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, gated=True):
        """
        Params:
            x: [bs x H x N x T]
        """
        x = self.conv1(x)
        if gated:
            ret = x[:,:self.c_out,:,:] * self.sigmoid(x[:,self.c_out:,:,:])
        else:
            ret = x[:,:self.c_out,:,:]
        return ret
