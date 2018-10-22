import torch
import numpy as np

def aveFracE(pred, y):
    return torch.mean(torch.abs((pred - y)/y))
