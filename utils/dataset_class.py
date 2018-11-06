import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import sys

class MyDataset(Dataset):
    def __init__(self, filename):
        #Initialize epoch_len
        #Identify where the data is, and store as object variable
        self.df = np.genfromtxt(filename, delimiter=',')
        self.epoch_len = len(self.df)
        self.feature_len = len(self.df[0]) - 1

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        #Return a dict with the sample, like the code snippet below:
        sample = {"x": self.df[idx][1:], 'y': self.df[idx][0]}
        return sample
