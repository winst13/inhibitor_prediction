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

'''    
class MolDataset(Dataset):
    def __init__(self, filename):
        #Initialize epoch_len
        #Identify where the data is, and store as object variable
        self.df = np.load(filename).items()
        self.chemids = self.df.keys()
        self.epoch_len = len(self.chemids)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        #Return a dict with the sample, like the code snippet below:
        chemid = self.chemids[idx]
        sample = {"chemid": chemid, 'label': self.df[chemid][0], 'smiles' : self.df[chemid][1], 'adj' : self.df[chemid][2]}
        return sample
'''