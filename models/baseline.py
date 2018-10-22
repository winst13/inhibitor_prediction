import torch
from torch import nn
from torch.nn import functional as F

class BaselineModel(nn.Module):
    def __init__(self, data_len, drop_rate = 0):
        super(BaselineModel, self).__init__()
        
        self.drop_rate = drop_rate
        self.data_len = data_len
        self.features = nn.Sequential(
            nn.Linear(data_len, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1),
        )
        
    def forward(self, x):
        features = self.features(x)
        return features