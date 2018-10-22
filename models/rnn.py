import torch
from torch import nn
from torch.nn import functional as F

from .multilayer import FCLayer

class LSTMModel(nn.Module):
    def __init__(self, batch, num_wavenumbers = 50, num_curves = 3, drop_rate = 0, hidden_size = 50, num_layers = 1, 
                 bidirectional = True, hiddens = [500, 200]):
        super(LSTMModel, self).__init__()
        
        self.drop_rate = drop_rate
        self.num_wavenumbers = num_wavenumbers
        self.num_curves = num_curves
        self.hidden_size = hidden_size
        self.batch = batch
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        self.rnn = nn.LSTM(input_size = 1, hidden_size = hidden_size, num_layers = num_layers, dropout = drop_rate, 
                                bidirectional = bidirectional)
        self.h0 = nn.Parameter(torch.randn(num_layers*self.num_directions, batch, hidden_size), 
                                  requires_grad=True).type(torch.FloatTensor)
        self.c0 = nn.Parameter(torch.randn(num_layers*self.num_directions, batch, hidden_size), 
                                  requires_grad=True).type(torch.FloatTensor)
        
        hiddens.insert(0, hidden_size*self.num_directions*num_wavenumbers)
        hiddens.append(num_curves)
        self.fc = nn.Sequential(
            *[FCLayer(hiddens[i], hiddens[i+1]) for i in range(len(hiddens)-1)]
        )
        print (self.rnn, self.fc)
        
    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(2) #make shapes work
        outputs, _ = self.rnn(x, (self.h0, self.c0))
        outputs = outputs.transpose(0, 1).contiguous().view(self.batch, -1)
        features = self.fc(outputs)
        return features