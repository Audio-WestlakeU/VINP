
import torch
import torch.nn as nn
class separableConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,bias=False):
        super(separableConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                                    in_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=in_channels,
                                    bias=bias)
        self.pointwise = nn.Conv1d(in_channels,
                                    out_channels,
                                    1,
                                    1,
                                    0,
                                    1,
                                    1,
                                     bias=bias)
    def forward(self,x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

