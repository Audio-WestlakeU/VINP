import torch
import torch.nn as nn
import numpy as np
from .attention_non import MultiHeadAttention


class ATTBlock(nn.Module):
    def __init__(self, input_dim=257, embed_dim=256, outpu_dim=257, number_heads=4, dropout1=0,dropout2=0):
        super(ATTBlock,self).__init__()
        self.att = MultiHeadAttention(input_dim, embed_dim, outpu_dim, number_heads, dropout1,dropout2)
        self.bn = nn.BatchNorm1d(input_dim)
        self.downsample = nn.Conv1d(input_dim,outpu_dim,1) if input_dim != outpu_dim else None
        
    def forward (self, x):
        
        x_bn = self.bn(x).permute(0,2,1)
        out = self.att(x_bn,x_bn,x_bn).permute(0,2,1)
        res = x if self.downsample is None else self.downsample(x)
        
        return out+res