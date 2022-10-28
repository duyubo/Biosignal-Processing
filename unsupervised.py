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
from Baselines.WCL import *
from Baselines.PSL import *
from Backbone.Transformer_Backbone import *
from train import *
from augmentation import *
import dataset_pretrain as DP
import dataset as D

def get_args_parser():
    parser = argparse.ArgumentParser('Self-supervised Learning For EEG/EMG Signals', add_help=False)
    # Model Parameters
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone Model')
    parser.add_argument('--method', default='SimCLR', type=str, help='Method')
    parser.add_argument('--seq_length', default=646, type=int, help='images input size')
    parser.add_argument('--seed', default=0, type=int, help='pytorch random seed')
    parser.add_argument('--input_dim', type=int, default=64, help='input_dimension')
    parser.add_argument('--path', type=str, default='./', help='model path')
    parser.add_argument('--out_dim', type=int, default=128, help='backboone output dimension')
    # CNN Backbone Parameters
    parser.add_argument('--p1', type=float, default=0.1, help='dropout')
    parser.add_argument('--p2', type=float, default=0.1, help='dropout')
    parser.add_argument('--p3', type=float, default=0.1, help='dropout')
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
    # Projection Header Parameters
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='projection header hidden dimension')
    parser.add_argument('--projection_size', type=int, default=64, help='projection size')
    #Predictor Header Parameters
    parser.add_argument('--predictor_mlp_hidden_size', type=int, default=256, help='predictor hidden dimension size')
    # Contrastive Learning Parameters
    parser.add_argument('--final_dim', type=int, default=5, help='Final Dimension')
    parser.add_argument('--temperature', type=float, default=0.1, help='contrastive learning temperature')
    parser.add_argument('--la', type=float, default=0.1, help='PSL unsupervised weight')
    parser.add_argument('--n_views', type=int, default=2, help='number of views')
    parser.add_argument('--topk', type=int, default=2, help='topk neighbors')
    parser.add_argument('--pretrained', action='store_true', help='Pretrained or Not')
    # Training Parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--patience', default=8, type=int, help='Patience of Self Supervised Learning')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (absolute lr)')
    # Dataset
    parser.add_argument('--dataset', default='EEG109', type=str, help='Dataset Name')
    parser.add_argument('--dataset_path', default='/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109.npy', type=str, help='Dataset Path')
    parser.add_argument('--training_type', default='unsupervised', type=str, help='training Type')
    parser.add_argument('--data_ratio_train', type=float, default=0.1, help='Pretraining sample rate for training')
    parser.add_argument('--data_ratio_val', type=float, default=0.1, help='Pretraining sample rate for validation')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training Data Ratio')
    parser.add_argument('--train_ratio_all', type=float, default=0.8, help='Training Data Ratio All')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation Data Ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test Data Ratio')
    parser.add_argument('--labeled_data', type=float, default=1, help='Labeled Data Ratio')
    parser.add_argument('--labeled_data_all', type=float, default=1, help='Labeled Data Ratio All')
    parser.add_argument('--split_stride', default=100, type=int, help='data split stride')
    return parser

def main_unsupervised(args):
    methods = {
      'BYOL': BYOL,
      'MoCo': MoCo, 
      'CLOCS': CLOCS, # single model, multiple views
      'SimCLR': ResNetSimCLR,
      'MAE': MaskedAutoencoderViT, # auto regression
      'WCL': WCL,
      'PSL': PSL
    }

    models = {
      'CNN': CNNModel,
      'ViT': ViTModel,
    }

    dataset = {
      'EEG109': DP.EEG109_Dataset if args.method == 'PSL' else D.EEG109_Dataset,
      'EEG2a': DP.EEG2a_Dataset if args.method == 'PSL' else D.EEG2a_Dataset,
      'Cho2017': DP.Cho2017_Dataset if args.method == 'PSL' else D.Cho2017_Dataset,
      'Shin2017': DP.Shin2017_Dataset if args.method == 'PSL' else D.Shin2017_Dataset,
      'HaLT12':DP.HaLT12_Dataset if args.method == 'PSL' else D.HaLT12_Dataset,
    }
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    train_data = dataset[args.dataset](split_type='train', args = args)  
    val_data = dataset[args.dataset](split_type='val', args = args)  
    

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    if args.method == 'PSL':
        val_loader = DataLoader(val_data, batch_size = 2048, shuffle = True)
    else:
        val_loader = DataLoader(val_data, batch_size = args.batch_size, shuffle = True)
    backbone = models[args.backbone](args = args).to(device)
    model = methods[args.method](backbone = backbone, args = args).to(device)
    if args.pretrained:
        print("Load model from pretrained one !!!")
        model = torch.load(args.path + args.dataset + '_' + args.method + '_'+ args.backbone + '.pt')
   
    if args.method == 'BYOL':
        optimizer = torch.optim.AdamW(list(model.online_network.parameters()) + list(model.predictor.parameters()), lr = args.lr, betas = (0.9, 0.95))
    elif args.method == 'MoCo':
        optimizer = torch.optim.AdamW(list(model.base_encoder.parameters()) + list(model.predictor.parameters()), lr = args.lr, betas = (0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, betas = (0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.99)
    loss_curve = []
    best_loss = 100000
    save_num = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model = model, device = device, train_loader = train_loader, scheduler = scheduler, optimizer = optimizer, epoch = epoch, method = args.method, args = args)
        loss_eval = evaluate(model = model, device = device, val_loader = val_loader, method = args.method)
        print('eopch: ', epoch, 'val loss: ', loss_eval)
        loss_curve.append(loss_eval[1])
        if loss_eval[0] < best_loss:
            torch.save(backbone, args.path + args.dataset + '_' + args.method + '_'+ args.backbone + '_Backbone.pt')
            #torch.save(model, args.path + args.dataset + '_' + args.method + '_'+ args.backbone + '.pt')
            best_loss = loss_eval[0]
            save_num = epoch
        if epoch - save_num > args.patience:
            break
    print(loss_curve)
    print('')

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
  
    main_unsupervised(args)




