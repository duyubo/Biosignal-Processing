import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

"""Derived from https://github.com/danikiyasseh/CLOCS/blob/master/prepare_network.py"""

class CNNModel(nn.Module):
    def __init__(self, args):#nencoders, p1, p2, p3, c1, c2, c3, c4, k, s, l
        super(CNNModel, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.dropout1 = nn.Dropout(p=args.p1)  
        self.dropout2 = nn.Dropout(p=args.p2)  
        self.dropout3 = nn.Dropout(p=args.p3)
        seq_l = args.seq_length
        k = args.kernel
        s= args.stride
        
        seq_l = torch.floor((torch.floor(torch.tensor((seq_l - args.first_kernel) / args.first_stride) + 1) - 2)/2 + 1).item()
        for i in range(2):
          seq_l = torch.floor((torch.floor(torch.tensor((seq_l - k) / s) + 1) - 2)/2 + 1).item()
        print(seq_l)
        self.backbone = nn.Sequential(
            nn.Conv1d(args.input_dim,args.c2,args.first_kernel,args.first_stride),
            nn.BatchNorm1d(args.c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout1,
            nn.Conv1d(args.c2,args.c3,k,s),
            nn.BatchNorm1d(args.c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout2,
            nn.Conv1d(args.c3,args.out_dim,k,s),
            nn.BatchNorm1d(args.out_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout3, 
        )
        self.final = nn.Linear(int(seq_l), 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.final(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class Transpose(torch.nn.Module):
    def forward(self, x):
        #print(x.shape)
        x = x.transpose(1, 2)
        return x
class MLPHead_PSL(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead_PSL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            Transpose(),
            nn.BatchNorm1d(mlp_hidden_size),
            Transpose(),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
