import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    """ Swish activation by Google that works better for 40-50+ layer networks.
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(data=torch.ones(1))
        self.beta.requires_grad = True

    def forward(self, x):
        #beta = self.beta.expand_as(x)
        return x * F.sigmoid(self.beta * x)