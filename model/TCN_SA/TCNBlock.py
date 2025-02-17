
import torch
import torch.nn as nn
import numpy as np
from .BasicConv1d import BasicConv1d


class TCNBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation1,dilation2):
    super(TCNBlock, self).__init__()
    
    self.conv1 = BasicConv1d(n_inputs, n_outputs, kernel_size,
                                       stride=1, dilation=dilation1)
    self.conv2 = BasicConv1d(n_outputs, n_outputs, kernel_size,
                                       stride=1, dilation=dilation2)
    

    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    
    

  def forward(self, x):
    
    out = self.conv1(x)
    out = self.conv2(out)
    res = x if self.downsample is None else self.downsample(x)
    
    return out + res


