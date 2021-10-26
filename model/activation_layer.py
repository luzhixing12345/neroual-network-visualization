import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RELU(nn.Module):
    def forward(self,x):
        return F.relu(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class change_dim(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim

    def forward(self,x):
        return x.view(self.dim)