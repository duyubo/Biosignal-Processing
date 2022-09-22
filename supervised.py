
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
import argparse

from Baselines.MAE import *
from Backbone.Transformer_Backbone import *
from dataset import *
from Baselines.SimCLR import *
from Baselines.CLOCS import *
from Baselines.BYOL import *
from Baselines.MocoV3 import *
from Baselines.PSL import *
from train import *

class SupervisedCNN(nn.Module):
    def __init__(self, backbone, args):
        super(SupervisedCNN, self).__init__()
        self.encoder = backbone
        self.final_layer = nn.Linear(args.out_dim, args.final_dim)
        self.__init_params()

    def __init_params(self):
        torch.nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.final_layer(x.squeeze(-1))
        return x

class SupervisedViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, backbone, args):
        super(SupervisedViT, self).__init__()
        self.encoder = backbone
        img_size = self.encoder.image_size
        patch_size = self.encoder.patch_size
        self.length_attn = nn.Linear((img_size[0] * img_size[1])//(patch_size[0]* patch_size[1]), 1)
        self.header = nn.Linear(args.decoder_embed_dim, args.final_dim)
        self.__init_params()

    def __init_params(self):
        torch.nn.init.xavier_uniform_(self.header.weight)
        nn.init.constant_(self.header.bias, 0)
        torch.nn.init.xavier_uniform_(self.length_attn.weight)
        nn.init.constant_(self.length_attn.bias, 0)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.encoder.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        x = self.length_attn(x[:, 1:, :].permute(0, 2, 1))
        output = self.header(x.squeeze())
        return output

def get_args_parser():
    parser = argparse.ArgumentParser('Self-supervised Learning For EEG/EMG Signals', add_help=False)
    # Model Parameters
    parser.add_argument('--backbone', default='CNNModel', type=str, help='Backbone Model')
    parser.add_argument('--method', default='SimCLR', type=str, help='Method')
    parser.add_argument('--seq_length', default=646, type=int, help='images input size')
    parser.add_argument('--seed', default=0, type=int, help='pytorch random seed')
    parser.add_argument('--input_dim', type=int, default=64, help='input_dimension')
    parser.add_argument('--path', type=str, default='./', help='model path')
    parser.add_argument('--out_dim', type=int, default=128, help='backboone output dimension')
    # CNN Backbone Parameters
    parser.add_argument('--p1', type=float, default=0.1, help='dropout')
    parser.add_argument('--p2', type=float, default=0., help='dropout')
    parser.add_argument('--p3', type=float, default=0., help='dropout')
    parser.add_argument('--c2', type=int, default=128, help='channel number')
    parser.add_argument('--c3', type=int, default=256, help='channel number')
    parser.add_argument('--kernel', type=int, default=7, help='CNN kernel size')
    parser.add_argument('--stride', type=int, default=3, help='CNN stride')
    parser.add_argument('--first_kernel', type=int, default=1, help='AvgPool kernel size')
    parser.add_argument('--first_stride', type=int, default=1, help='AvgPool stride size')
    #ViT Backbone Parameters
    parser.add_argument('--depth', type=int, default=1, help='encoder number')
    parser.add_argument('--patch_x', type=int, default=323, help='patch size x')
    parser.add_argument('--patch_y', type=int, default=16, help='patch size y')
    parser.add_argument('--embed_dim', type=int, default=20, help='embedded dim')
    parser.add_argument('--num_heads', type=int, default=4, help='number of head')
    parser.add_argument('--decoder_embed_dim', type=int, default=20, help='embedded dim of decoder')
    parser.add_argument('--decoder_depth', type=int, default=2, help='depth of decoder')
    parser.add_argument('--decoder_num_heads', type=int, default=4, help='number of head of decoder')
    # Supervised Learning Parameters
    parser.add_argument('--final_dim', type=int, default=5, help='Final Dimension')
    parser.add_argument('--pretrained', action='store_true', help='Pretrained or Not')
    # Training Parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--patience', default=20, type=int, help='Patience of Self Supervised Learning')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (absolute lr)')
    # Dataset
    parser.add_argument('--dataset', default='EEG109', type=str, help='Dataset Name')
    parser.add_argument('--dataset_path', default='/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109.npy', type=str, help='Dataset Path')
    parser.add_argument('--training_type', default='supervised', type=str, help='training Type')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training Data Ratio')
    parser.add_argument('--train_ratio_all', type=float, default=0.8, help='Training Data Ratio All')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation Data Ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test Data Ratio')
    parser.add_argument('--labeled_data', type=float, default=0.1, help='Labeled Data Ratio')
    parser.add_argument('--labeled_data_all', type=float, default=1, help='Labeled Data Ratio All')
    parser.add_argument('--split_stride', default=100, type=int, help='data split stride')
    
    return parser

def main_supervised(args):
    print(args)
    args.train_ratio_all = args.train_ratio
    args.labeled_data_all = args.labeled_data
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = args.method

    backbone_model = {
      'CNN': CNNModel,
      'ViT': ViTModel,
    }

    supervised_model = {
      'CNN': SupervisedCNN,
      'ViT': SupervisedViT,
    }

    dataset = {
      'EEG109': EEG109_Dataset,
      'EEG2a':EEG2a_Dataset,
      'EMGDB7': EMGDB7_Dataset,
    }

    train_data = dataset[args.dataset](split_type = 'train', args = args)  
    val_data = dataset[args.dataset](split_type = 'val', args = args) 
    test_data = dataset[args.dataset](split_type = 'test', args = args)
    
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)

    backbone = backbone_model[args.backbone](args)
    print(args.pretrained)
    if args.pretrained:
        print("Load model from pretrained one !!!")
        backbone = torch.load(args.path + args.dataset + '_' + args.method + '_'+ args.backbone + '_Backbone.pt')
    model = supervised_model[args.backbone](backbone, args).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    lr = args.lr  # learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)
    loss_curve = []
    best_val_acc = 0
    save_num = 0
    if args.pretrained:
        saved_name = args.path + args.dataset + '_'+ args.method + '_' + args.backbone + '_' + str(args.labeled_data) + 'Pretrained_' + str(args.pretrained) + '.pt'
    else:
        saved_name = args.path + args.dataset + '_' + args.backbone + '_' + str(args.labeled_data) + 'Pretrained_' + str(args.pretrained) + '.pt'
    for epoch in range(0, args.epochs):
        loss = train_supervised(model = model, device = device, train_loader = train_loader, 
                            scheduler = scheduler, method = method, criterion = criterion, 
                            optimizer = optimizer, epoch = epoch)
        val_acc = evaluate_supervised(model = model, train_loader = val_loader, method = method, device = device)
        loss_curve.append(val_acc)
        print('epoch', val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, saved_name)
            save_num = epoch
        if epoch - save_num > args.patience:
            break
    print(loss_curve)
    print('')

    model = torch.load(saved_name)
    test_loss = evaluate_supervised(model = model, train_loader = test_loader, method = method, device = device)
    print('best validation loss: ', best_val_acc, 'test loss: ', test_loss)
    return best_val_acc, test_loss
    
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    results_summary = []
    main_supervised(args)
    """for labeled_ratio in [1]:# 0.01, 0.1, 0.2, 0.4, 0.6, 0.8,
        args.labeled_data = labeled_ratio
        results_summary.append(main_supervised(args))"""
    """for train_ratio in [ 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:#
        args.train_ratio = train_ratio
        results_summary.append(main_supervised(args))"""
    print(results_summary)
    
    
    



