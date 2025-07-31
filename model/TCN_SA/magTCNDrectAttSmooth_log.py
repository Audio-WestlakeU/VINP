import torch
import torch.nn as nn
import numpy as np
from .ATTBlock import ATTBlock
import torch.nn.functional as F
from .TCN import TCN
from .BasicConv1d import SmoothConv1d


class magTCNDrectAttSmooth(nn.Module):
    def __init__(
        self,
        input_size=257,
        output_size=257,
        num_channels=[512, 512] * 4,
        kernel_size=3,
        repeat=[1, 2, 5, 9] * 4,
        num_heads=4,
        embed_size=256,
        dropout1=0,
        dropout2=0,
    ):

        super(magTCNDrectAttSmooth, self).__init__()
        self.att = ATTBlock(
            input_size, embed_size, output_size, num_heads, dropout1, dropout2
        )
        self.tcn = TCN(input_size, num_channels, kernel_size, repeat)
        self.lin = nn.Linear(num_channels[-1], output_size)
        self.smo = SmoothConv1d(
            output_size, output_size, kernel_size, stride=1, dilation=1
        )

    def forward(self, x):

        B_dim,C_dim,F_dim,T_dim = x.shape

        assert C_dim == 1
        
        x = x.reshape(B_dim,F_dim,T_dim)
        
        xx = self.att(x)

        xx = self.tcn(xx)

        xx = xx.transpose(1, 2)

        xx = self.lin(xx)

        xx = self.smo(xx.transpose(1, 2)).transpose(1, 2)

        # y = F.relu(xx, inplace=False)


        y = xx.permute(0,2,1).unsqueeze(1)

        return y
