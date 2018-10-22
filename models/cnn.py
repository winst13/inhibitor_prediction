import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .multilayer import FCLayer, VariableModel

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, drop_rate = 0, activation_func = nn.LeakyReLU):
        super(ConvLayer, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = (kernel_size-1)//2),
            nn.Dropout(p = drop_rate),
            activation_func()
        )
    
    def forward(self, x):
        return self.layers(x)

class ConvModel(nn.Module):
    def __init__(self, num_wavenumbers = 1011, num_curves = 5, drop_rate = 0, channels = [16, 16, 16, 16, 32], 
                 kernel_sizes = [7, 5, 5, 3, 3], strides = [1, 2, 2, 2, 1], hiddens = [100, 200], conv_out_size = 127, 
                 activation_func = nn.LeakyReLU):
        super(ConvModel, self).__init__()
        assert len(channels) == len(kernel_sizes) and len(kernel_sizes) == len(strides)
        
        self.drop_rate = drop_rate
        self.num_wavenumbers = num_wavenumbers
        self.num_curves = num_curves
        self.num_conv = len(channels)
        self.conv_out_params = conv_out_size*channels[-1]
        
        channels.insert(0, 1)
        hiddens.insert(0, self.conv_out_params)
        hiddens.append(num_curves)
        
        self.conv = nn.Sequential(
            *[ConvLayer(channels[i], channels[i+1], kernel_sizes[i], strides[i], drop_rate=drop_rate,
                        activation_func = activation_func) for i in range(self.num_conv)]
        )
        
        self.fc = nn.Sequential(
            *[FCLayer(hiddens[i], hiddens[i+1], activation_func = activation_func) for i in range(len(hiddens)-1)]
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) #channels_in = 1, doing this so shapes make sense
        out = self.conv(x)
        out = out.view(-1, self.conv_out_params)
        features = self.fc(out)
        return features