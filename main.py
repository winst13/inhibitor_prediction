import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.datasets as dset
import torchvision.transforms as T
from torch.nn import functional as F

from tensorboardX import SummaryWriter

import numpy as np
import argparse
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from utils.dataset_class import *
from models.baseline import BaselineModel
from models.multilayer import MultilayerModel, VariableModel
from models.cnn import ConvModel
from utils.checkpoint import save_model, load_model
from utils.scheduler import CosineAnnealingWithRestartsLR
from models.swish import Swish

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action='store_true', help="Debug mode: ???")
parser.add_argument("--num_epochs", default = 0, type = int, help = "optional argument, add if you want to stop after #epochs")
parser.add_argument("--load_check", action='store_true', help="optional argument, if we want to load an existing model")
parser.add_argument("--load_best", action='store_true', help="optional argument, if we want to load an existing model")
parser.add_argument("--print_every", default = 10, type=int, help="print loss every this many iterations")
parser.add_argument("--use_cpu", action='store_true', help="Use GPU if possible, set to false to use CPU")
parser.add_argument("--batch_size", default = 100, type=int, help="Batch size")
parser.add_argument("--val_ratio", default = 0.2, type=float, help="Proportion of training data set aside as validation")
parser.add_argument("--mode", help="can be train, test, or vis")
parser.add_argument("--save_every", default = 1, type=int, help="save model at this this many epochs")
parser.add_argument("--model_file", help="mandatory argument, specify which file to load model from")
parser.add_argument("--exp_name", help="mandatory argument, specify the name of the experiment")
parser.add_argument("--model", help="mandatory argument, specify the model being used")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--dropout", default=0, type=float, help="dropout rate.  higher = more dropout")
parser.add_argument("--l2reg", default=0, type=float, help="l2 regularization rate")
parser.add_argument("--epoch_len", default = 10000, type=int, help="Number of curves in an epoch")
parser.add_argument("--optimizer", default="Adam", help="type of optimizer")
parser.add_argument("--scheduler", default=None, help="tells the model what kind of scheduler to create")
parser.add_argument("--generator_mode", default="default", help="what kind of curves to generate, also subfolder name")
parser.add_argument("--loss_name", default="MSE", help="what kind of loss function to use")
parser.add_argument("--momentum", default=-1.0, type=float, help="momentum of optimizer, if it requires one")
parser.add_argument("--nesterov", action='store_true', help="whether SGD is Nesterov or not")
args = parser.parse_args()

#Setup
debug = args.debug
USE_GPU = False if args.use_cpu else True
VAL_RATIO = args.val_ratio
num_epochs = args.num_epochs
mode = args.mode
model_name = args.model
optimizer_name = args.optimizer
learning_rate = args.lr
dropout = args.dropout
l2reg = args.l2reg
scheduler_name = args.scheduler
generator_mode = args.generator_mode
loss_name = args.loss_name
momentum = args.momentum
nesterov = args.nesterov

BATCH_SIZE = args.batch_size
EPOCH_LEN = args.epoch_len

validation_split = .2
shuffle_dataset = True
random_seed= 42

#Tensorboard and saving/loading
load_check = args.load_check
load_best = args.load_best
print_every = args.print_every
save_every = args.save_every
exp_name = args.exp_name
exp_dir = os.path.join("experiments", generator_mode)

#Make sure parameters are valid
assert mode == 'train' or mode == 'val' or mode == 'test'
assert load_check == False or load_best == False
assert exp_name is not None
if mode == 'val'or mode == 'test':  
    assert load_check == True or load_best == True
    
dtype = torch.float32
if USE_GPU and torch.cuda.is_available(): #Determine whether or not to use GPU
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

#define data sets and data loaders
data_len = None
input_len = None
if generator_mode == "default":
    generator_train = MyDataset(filename = "./data/train_test/cdk2_train.csv")
    generator_test = MyDataset(filename = "./data/train_test/cdk2_test.csv")
    data_len = len(generator_train)
    input_len = generator_train.feature_len
    indices = list(range(data_len))
    split = int(np.floor(validation_split * data_len))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    loader = DataLoader(generator_train, batch_size=BATCH_SIZE, sampler = train_sampler)
    loader_val = DataLoader(generator_train, batch_size=BATCH_SIZE, sampler = val_sampler)
    loader_test = DataLoader(generator_test, batch_size=BATCH_SIZE)
    
#create models
if model_name == "baseline":
    model = BaselineModel(input_len, drop_rate=dropout)
else:
    sys.exit("model_name %s was not found"%(model_name))
    
#create optimizers
optimizer = None
if optimizer_name == "Adam":
    betas = (0.9, 0.999)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, 
                                 betas = betas, weight_decay=l2reg)
elif optimizer_name == "RMSprop":
    assert momentum > -1.0
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, 
                        momentum=momentum, weight_decay=l2reg)
elif optimizer_name == "SGD":
    assert momentum > -1.0
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum,
                                weight_decay=l2reg, nesterov=nesterov)
elif optimizer_name == "Adagrad":
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, 
                                    weight_decay=l2reg)
else: 
    sys.exit("optimizer_name %s was not found"%(optimizer_name))

#learning rate decay
scheduler = None
if scheduler_name is not None:
    if scheduler_name == "rop":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 25, factor = 0.2, verbose=True)
    elif scheduler_name == "cos":
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=100)
    else:
        sys.exit("scheduler name %s was not found"%(scheduler_name))
    
#loss function for test (always MSE loss for training)
loss_func = None
if loss_name == "MSE":
    loss_func = nn.MSELoss()
elif loss_name == "aveFracE":
    loss_func = aveFracE
else:
    sys.exit("loss_func name %s was not found"%(loss_name))

#method for training
def train(train_loader, val_loader, model, optimizer, writer, debug, epoch = 0, scheduler = None, 
          test_loss_func = nn.MSELoss, max_epoch = 0, ):
    while True:
        for t, sample in enumerate(train_loader):
            #get x, y out of sample
            x = sample['x']
            y = sample['y']
            
            # Move the data to the proper device (GPU or CPU)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.float)
            
            pred = model(x).squeeze()
            optimizer.zero_grad()
            loss = nn.MSELoss()(pred, y) #may need to change loss function
            loss.backward()
            optimizer.step()
            
            if t % print_every == 0:
                print('Iteration %d: batch train loss = %06f'%(t, float(loss.item())))
        val_loss, val_acc = test(val_loader, model, loss_func, debug)
        
        if scheduler is not None:
            scheduler.step(test_loss)
        print ('Epoch %d: train loss = %06f'%(epoch, float(loss.item())))
        print ('Epoch %d: val loss = %06f'%(epoch, val_loss))
        print ('Epoch %d: val acc = %06f'%(epoch, val_acc))
        if epoch%save_every == 0:
            writer.add_scalar('loss', val_loss, epoch)
            save_model({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, val_loss, exp_dir, exp_name)
        if epoch == 0:
            writer.add_graph(model, x)
        epoch += 1
        if max_epoch != 0 and epoch >= max_epoch:
            break

#method for testing
def test(loader, model, loss_func, debug):
    loss = 0
    model.eval()  # set model to evaluation mode
    truepos = 0
    falsepos = 0
    trueneg = 0
    falseneg = 0
    tot = 0
    with torch.no_grad():
        for t, sample in enumerate(loader):
            x = sample['x']
            y = sample['y']
            
            # Move the data to the proper device (GPU or CPU)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.float)
            
            pred = model(x).squeeze()
            loss += loss_func(pred, y)
            
            truepos += len([1 for i in range(len(y)) if y[i] == 1 and torch.round(pred[i]) == 1])
            falsepos += len([1 for i in range(len(y)) if y[i] == 0 and torch.round(pred[i]) == 1])
            trueneg += len([1 for i in range(len(y)) if y[i] == 0 and torch.round(pred[i]) == 0])
            falseneg += len([1 for i in range(len(y)) if y[i] == 1 and torch.round(pred[i]) == 0])
            tot += len(pred)

            if debug:
                pass
                #do something
        
        print (truepos, falsepos, trueneg, falseneg)
        accuracy = (truepos + trueneg)/tot
        precision = truepos/(truepos + falsepos)
        recall = truepos/(truepos + falseneg)
        f1 = 2*precision*recall/(precision + recall)
        print ("F1 = ", f1)
    return loss, accuracy #may or may not want to add other things to return

epoch = 0
if load_check:
    epoch = load_model(exp_dir, exp_name, model, optimizer, mode = 'checkpoint', lr = learning_rate)
elif load_best:
    epoch = load_model(exp_dir, exp_name, model, optimizer, mode = 'best', lr = learning_rate)
    
#execute something
try:
    if mode == 'train':
        writer = SummaryWriter(log_dir=os.path.join(exp_dir, exp_name))
        train(loader, loader_val, model, optimizer, writer, debug,
              epoch = epoch, test_loss_func = loss_func, max_epoch = num_epochs, scheduler=scheduler)
    elif mode == 'test':
        loss = test(loader_test, model, debug, loss_func = loss_func)
        print ("test loss = %06f"%(loss))
        
except KeyboardInterrupt:
    print ('Interrupted')
    if mode == 'train':
        writer.close()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
