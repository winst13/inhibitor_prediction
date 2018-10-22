import torch
from torch import nn
from torch.nn import functional as F

class MultilayerModel(nn.Module):
    def __init__(self, num_wavenumbers = 50, num_curves = 3, drop_rate = 0, hidden1 = 500, hidden2 = 500, 
                 activation_func = nn.LeakyReLU):
        super(MultilayerModel, self).__init__()
        
        self.drop_rate = drop_rate
        self.num_wavenumbers = num_wavenumbers
        self.num_curves = num_curves
        self.features = nn.Sequential(
            nn.Linear(num_wavenumbers, hidden1), 
            activation_func(),
            nn.Linear(hidden1, hidden2), 
            activation_func(),
            nn.Linear(hidden2, num_curves),
            activation_func()
        )
        
    def forward(self, x):
        features = self.features(x)
        return features

class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0, activation_func = nn.LeakyReLU):
        super(FCLayer, self).__init__()
        
        self.drop_rate = drop_rate
        self.features = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            activation_func()
        )
        
    def forward(self, x):
        features = self.features(x)
        return features
    
class BNormFCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0, activation_func = nn.LeakyReLU):
        super(BNormFCLayer, self).__init__()
        
        self.drop_rate = drop_rate
        self.features = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim), 
            activation_func()
        )
        
    def forward(self, x):
        features = self.features(x)
        return features

class VariableModel(nn.Module):
    def __init__(self, num_wavenumbers = 50, num_curves = 3, drop_rate = 0, hiddens = [500, 500, 500], batchnorm=False, 
                 activation_func = nn.LeakyReLU):
        super(VariableModel, self).__init__()
        
        self.drop_rate = drop_rate
        self.num_wavenumbers = num_wavenumbers
        self.num_curves = num_curves
        hiddens.insert(0, num_wavenumbers)
        hiddens.append(num_curves)
        if batchnorm:
            self.features = nn.Sequential(
                *[BNormFCLayer(hiddens[i], hiddens[i+1], activation_func = activation_func) for i in range(len(hiddens)-1)]
            )
        else:
            self.features = nn.Sequential(
                *[FCLayer(hiddens[i], hiddens[i+1], activation_func = activation_func) for i in range(len(hiddens)-1)]
            )
        
    def forward(self, x):
        features = self.features(x)
        return features