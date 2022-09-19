import torchaudio
import torchaudio.transforms as T

from typing import Tuple
import copy
import time

from torch.utils.data import dataset
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import librosa
import argparse

from Baselines.MAE import *
from Baselines.SimCLR import *
from Baselines.CLOCS import *
from Baselines.BYOL import *
from Baselines.MocoV3 import *
from Baselines.PSL import *
from Backbone.Transformer_Backbone import *
from dataset import *
from augmentation import *

def adjust_learning_rate(warm_up, optimizer, epoch, base_lr, i, iteration_per_epoch, epochs):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model: nn.Module, device, train_loader, scheduler, method, optimizer, epoch, args, moco_m = 0.996):
    model.train()  # turn on train mode
    total_loss = 0.
    start_time = time.time()
    batch_num = 0
    log_interval = 100
    pseudo_loss = 0
    for batch, (batch_X, batch_Y) in enumerate(train_loader):
        data = batch_X.to(device)
        if method != 'MAE':
            data = data.squeeze(1)
            data1 = []
            data2 = []
            for d in data:
                #data1.append(stretch_spec(d, device, 1.1))
                #data2.append(stretch_spec(d, device, 1.1))
                #data1.append(mask_freq_spec(d, device, 0.2))
                #data2.append(mask_freq_spec(d, device, 0.2))
                #print(mask_freq_spec(d, device, 0.2).shape)
                #data1.append(gaussian_noise(mask_freq_spec(d, device, 0.2), 0.5))
                #data2.append(gaussian_noise(mask_freq_spec(d, device, 0.2), 0.5))
                data1.append(gaussian_noise(d, 1).transpose(0, 1))
                data2.append(gaussian_noise(d, 1).transpose(0, 1))
            data1 = torch.stack(data1)
            data2 = torch.stack(data2)
        if method == 'SimCLR':
            data = torch.cat([data1, data2])
            data = data.to(device)
            output = model(data, torch.arange(batch_X.shape[0]).to(device))
        elif method == 'MAE':
            output = model(data, 0.05) 
        elif method == 'CLOCS' or method == 'PSL':
            data = torch.cat([data1, data2])
            data = data.to(device)
            output = model(data, batch_Y)
        elif method == 'BYOL' or method == 'WCL':
            if method == 'WCL' and epoch <= 10:
                adjust_learning_rate(warm_up = 10, optimizer = optimizer, epoch = epoch, base_lr = args.lr, i = batch, iteration_per_epoch = len(train_loader), epochs = args.epochs)
            batch_view_1 = data1.to(device)
            batch_view_2 = data2.to(device)
            if method == 'WCL':
                output = model(batch_view_1, batch_view_2, p_labels = batch_Y, epoch = epoch)
            else:
                output = model(batch_view_1, batch_view_2)
        elif method == 'MoCo':
            batch_view_1 = data1.to(device)
            batch_view_2 = data2.to(device)
            output = model(batch_view_1, batch_view_2, moco_m)
        else:
            raise NotImplementedError('no such ' + method)
        loss = output[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if method == 'WCL':
            pseudo_loss += output[3].item()
        if method == 'BYOL':
            model._update_target_network_parameters()
        total_loss += loss.item()
        batch_num += 1

    lr = scheduler.get_last_lr()[0]
    ms_per_batch = (time.time() - start_time) * 1000 / batch_num
    loss = total_loss / batch_num
    if method == 'WCL':
        print('pseudo label acc: ', pseudo_loss/batch_num)
    print(f'| epoch {epoch:3d} | {batch:5d} batches | '
          f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {loss:5.2f}')
    return loss, pseudo_loss/batch_num

def evaluate(model: nn.Module, device, val_loader, method, moco_m = 0.996) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    batch_num = 0
    pseudo_loss = 0
    with torch.no_grad():
        for batch, (batch_X, batch_Y) in enumerate(val_loader):
            data = batch_X.to(device)
            if method != 'MAE':
                data = data.squeeze(1)
                data1 = []
                data2 = []
                for d in data:
                    #data1.append(stretch_spec(d, device, 1.1))
                    #data2.append(stretch_spec(d, device, 1.1))
                    #data1.append(mask_freq_spec(d, device, 0.2))
                    #data2.append(mask_freq_spec(d, device, 0.2))
                    #data1.append(gaussian_noise(mask_freq_spec(d, device, 0.2), 0.5))
                    #data2.append(gaussian_noise(mask_freq_spec(d, device, 0.2), 0.5))
                    data1.append(gaussian_noise(d, 1).transpose(0, 1))
                    data2.append(gaussian_noise(d, 1).transpose(0, 1))
                data1 = torch.stack(data1)
                data2 = torch.stack(data2)
            if method == 'SimCLR':
                data = torch.cat([data1, data2])
                data = data.to(device)
                output = model(data, torch.arange(batch_X.shape[0]).to(device))
            elif method == 'MAE':
                output = model(data, 0.05) 
            elif method == 'CLOCS' or method == 'PSL':
                data = torch.cat([data1, data2])
                data = data
                output = model(data, batch_Y) 
            elif method == 'BYOL' or method == 'WCL':
                batch_view_1 = data1.to(device)
                batch_view_2 = data2.to(device)
                if method == 'WCL':
                    output = model(batch_view_1, batch_view_2, p_labels = batch_Y)
                else:
                    output = model(batch_view_1, batch_view_2)
            elif method == 'MoCo':
                batch_view_1 = data1.to(device)
                batch_view_2 = data2.to(device)
                output = model(batch_view_1, batch_view_2, moco_m)
            else:
                raise NotImplementedError('no such ' + method)
            loss = output[0]
            total_loss += loss.item()
            if loss.item() > 0:
                batch_num += 1
            if method == 'WCL':
                pseudo_loss += output[3].item()
    if method == 'WCL':
        print('validation pseudo label acc: ', pseudo_loss/batch_num)
    return total_loss/batch_num, pseudo_loss/batch_num

def train_supervised(model, device, train_loader, scheduler, method, criterion, optimizer, epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    start_time = time.time()
    batch_num = 0
    for batch, (batch_X, batch_Y) in enumerate(train_loader):
        data = batch_X.to(device)
        if method != 'MAE':
            data = data.squeeze(1).transpose(1, 2)
        targets = batch_Y.to(device)
        output = model(data)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1

    lr = scheduler.get_last_lr()[0]
    ms_per_batch = (time.time() - start_time) * 1000 / batch_num
    loss = total_loss / batch_num
    print(f'| epoch {epoch:3d} | {batch:5d} batches | '
          f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {loss:5.2f}')
    return loss
     
def evaluate_supervised(model, train_loader, method, device) -> float:
    model.eval()  # turn on evaluation mode
    total_length = 0
    correct = 0
    with torch.no_grad():
        for batch, (batch_X, batch_Y) in enumerate(train_loader):
            data = batch_X.to(device)
            if method != 'MAE':
                data = data.squeeze(1).transpose(1, 2)
            targets = batch_Y.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_length += targets.shape[0]
            correct += (predicted == targets).sum().item()
    return correct / total_length


