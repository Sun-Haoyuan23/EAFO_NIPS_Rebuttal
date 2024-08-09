import torch
import torch.nn as nn
import torch.nn.functional as F

class StarReLU(nn.Module):
    def __init__(self, s=0.8944, b=-0.4472):
        super(StarReLU, self).__init__()
        self.s=nn.Parameter(torch.tensor(s))
        self.b=nn.Parameter(torch.tensor(b))
    def forward(self,x):
        return self.s*(F.relu(x))**2 + self.b
