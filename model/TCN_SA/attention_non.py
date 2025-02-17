import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer
    """
    def __init__(self, input_dim=257, embed_dim=256, output_dim=257, num_heads=4, dropout1=0.0, dropout2=0.0):
        """
        Parameters:
        input_dim: size of input
        embed_dim: size of embedding
        output_dim: size of output
        num_heads: number of attention heads
        dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** (-0.5)

        self.query_lin = nn.Linear(input_dim, embed_dim, bias=False)
        self.key_lin = nn.Linear(input_dim, embed_dim, bias=False)
        self.value_lin = nn.Linear(input_dim, embed_dim, bias=False)
        self.output_lin = nn.Linear(embed_dim, output_dim, bias=False)

        self.dp1 = nn.Dropout(dropout1)
        self.dp2 = nn.Dropout(dropout2)

    def _split_heads(self, x):
        """
        Input:
        x (batch_size x seq_len x dim)
        Return:
        a tensor (batch_size x num_heads x seq_len x dim/num_heads)
        """
        shape = x.shape
        if len(shape) != 3:
            raise ValueError("sequence should be NxTxD")
        return x.view(shape[0], shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Input:
        x (batch_size x num_heads x seq_len x dim/num_heads)
        Return:
        a tensor (batch_size x seq_len x dim)
        """
        shape = x.shape
        if len(shape) != 4:
            raise ValueError("sequence should be NxHxTxHD")
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], self.embed_dim)

   
    
    def forward(self, queries, keys, values):
        """
        compute multi-head attention
        """
        max_len = queries.shape[1]

            
        # linear transform
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        # split to multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # scale queries
        queries *= self.scaling

        # dot mutiplication
        qk = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        # pass through softmax
        w = F.softmax(qk, dim=-1)

        # dropout
        w = self.dp1(w)

        # get contexts
        contexts = torch.matmul(w, values)

        # merge heads
        contexts = self._merge_heads(contexts)

        # embed to output
        outputs = self.output_lin(contexts)

        return self.dp2(outputs)

        
