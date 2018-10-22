import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import pandas as pd
import sys

DATA_DIR = "./spectra"

class MyDataset(Dataset):
    def __init__(self, filename = ""):
        #Initialize epoch_len
        #Identify where the data is, and store as object variable
        pass

    def __len__(self):
        pass
        #return self.epoch_len

    def __getitem__(self, idx):
        #Return a dict with the sample, like the code snippet below:
        #sample = {"x": self.x, 'y': self.y}
        #return sample
        pass
