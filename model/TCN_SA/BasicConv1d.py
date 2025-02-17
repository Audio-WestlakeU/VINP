
import torch
import torch.nn as nn
from .separableConv1d import separableConv1d

class BasicConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,dilation):
        super(BasicConv1d, self).__init__()
        p=(kernel_size-1)*dilation
        self.conv = separableConv1d(in_channels,out_channels,kernel_size,stride,p,dilation,bias=False)
        self.bn=nn.BatchNorm1d(in_channels)
        self.chomp=Chomp1d(int(p/2))  if p!=0 else None
        self.prelu=nn.PReLU()

    def forward(self,x):
        
        x = self.conv(self.prelu(self.bn(x)))
        if self.chomp is not None:
                
                x=self.chomp(x)
                
        return x

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, self.chomp_size:-self.chomp_size].contiguous()


class SmoothConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,dilation=1):
        super(SmoothConv1d, self).__init__()
        p=(kernel_size-1)*dilation
        self.conv = separableConv1d(in_channels,out_channels,kernel_size,stride,p,dilation,bias=False)
        self.chomp=Chomp1d(int(p/2))  if p!=0 else None

    def forward(self,x):
        
        x = self.conv(x)
        if self.chomp is not None:
                
                x=self.chomp(x)
                
        return x