import torch
from torch import nn
from torch.nn import functional as F

class BaselineModel(nn.Module):
    def __init__(self, input_len, drop_rate = 0):
        super(BaselineModel, self).__init__()
        
        self.drop_rate = drop_rate
        self.input_len = input_len
        self.features = nn.Sequential(
            nn.Linear(self.input_len, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 250),
            nn.LeakyReLU(),
            nn.Linear(250, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        features = self.features(x)
        return features