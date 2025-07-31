

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .TCNBlock import TCNBlock
import torch.nn.init as winit
from .separableConv1d import separableConv1d

class TCN(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=3, repeat=[1,2,5,9]*4):
    super(TCN, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
        d1 = repeat[2*i] 
        d2 = repeat[2*i+1] 
        in_channels = num_inputs if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
    
        layers += [TCNBlock(in_channels, out_channels, kernel_size, 1, d1,d2)]
      
    self.network = nn.Sequential(*layers)

    
    
  def forward(self, x):
       


        return self.network(x)

# class AttentionBlock(nn.Module):
#   """An attention mechanism similar to Vaswani et al (2017)
#   The input of the AttentionBlock is `BxTxD` where `B` is the input
#   minibatch size, `T` is the length of the sequence `D` is the dimensions of
#   each feature.
#   The output of the AttentionBlock is `BxTx(D+V)` where `V` is the size of the
#   attention values.
#   Arguments:
#       dims (int): the number of dimensions (or channels) of each element in
#           the input sequence
#       k_size (int): the size of the attention keys
#       v_size (int): the size of the attention values
#       seq_len (int): the length of the input and output sequences
#   """
#   def __init__(self, dims, k_size, v_size, seq_len=None):
#     super(AttentionBlock, self).__init__()
#     self.key_layer = nn.Linear(dims, k_size)
#     self.query_layer = nn.Linear(dims, k_size)
#     self.value_layer = nn.Linear(dims, v_size)
#     self.sqrt_k = math.sqrt(k_size)

#   def forward(self, minibatch):
#     keys = self.key_layer(minibatch)
#     queries = self.query_layer(minibatch)
#     values = self.value_layer(minibatch)
#     logits = torch.bmm(queries, keys.transpose(2,1))
#     # Use numpy triu because you can't do 3D triu with PyTorch
#     # TODO: using float32 here might break for non FloatTensor inputs.
#     # Should update this later to use numpy/PyTorch types of the input.
#     #mask = np.triu(np.ones(logits.size()), k=1).astype('uint8')
#     #mask = torch.from_numpy(mask).cuda()
#     # do masked_fill_ on data rather than Variable because PyTorch doesn't
#     # support masked_fill_ w/-inf directly on Variables for some reason.
#     #logits.data.masked_fill_(mask, float('-inf'))
#     probs = F.softmax(logits, dim=1) / self.sqrt_k
#     read = torch.bmm(probs, values)
#     return minibatch + read
