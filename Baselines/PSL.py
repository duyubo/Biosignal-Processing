import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from Backbone.CNN_Backbone import *

class PSL(nn.Module):
    def __init__(self, backbone, args):
        super(PSL, self).__init__()
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.backbone = backbone
        self.projection = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
        self.diff =  nn.MultiheadAttention(args.projection_size, 16, dropout=0.1)
        self.header = nn.Linear(args.projection_size, args.final_dim * (args.final_dim -1) // 2 + 1)
        x_c, y_c = torch.meshgrid(torch.arange(args.final_dim), torch.arange(args.final_dim))
        self.map_indexes = torch.stack([x_c.triu(diagonal = 1), y_c.triu(diagonal = 1)]).permute(1, 2, 0).cuda()
        #print(self.map_indexes)
        self.final_dim = args.final_dim
    def info_nce_loss(self, features, labels):
        batch_size = labels[0].shape[0]
        seq_l = features.shape[2]
        p_x, p_y = torch.meshgrid(labels[1], labels[1])
        p_binary_metrix = (p_x == p_y).cuda()

        l_x, l_y = torch.meshgrid(labels[0], labels[0])
        l_binary_metrix = (l_x != l_y).cuda()
        l_metrix = torch.stack([l_x, l_y]).permute(1, 2, 0).cuda()

        equal_label_metrix = (l_x == l_y).cuda()

        i_x, i_y = torch.meshgrid(torch.arange(batch_size), torch.arange(batch_size))
        i_metrix = torch.stack([i_x, i_y]).permute(1, 2, 0).cuda()
        
        l_metrix[equal_label_metrix] = 0

        #similarity could be replaced by other function or w
        binary_metrix = p_binary_metrix
        #binary_metrix = torch.logical_and(p_binary_metrix, l_binary_metrix)
        labels_difference = l_metrix[binary_metrix]
        select_indexes = i_metrix[binary_metrix]
        batch_size = select_indexes.shape[0]
        features = features.index_select(dim = 0, index = select_indexes.reshape(-1))#
        features = features.reshape(batch_size, 2, features.shape[1], features.shape[2]).permute(1, 0, 3, 2)
        #features.shape = 2, batch_size, seq_length, hidden_dim

        diff_metrix, _ = self.diff(query = features[0], key = features[1], value = features[1], need_weights=False)
        diff_metrix = self.backbone.final(diff_metrix.transpose(1, 2))
        diff_metrix = diff_metrix.squeeze(-1)
        # get class pair (class i and class j, i!= j) from same subject
        unique_pair, new_label = torch.unique(torch.cat([self.map_indexes.reshape(-1, 2), labels_difference.sort()[0]]), dim = 0, return_inverse = True)
        diff_metrix = self.header(diff_metrix)
        weights = []
        for i in range(unique_pair.shape[0]):
            counter = ((new_label[self.final_dim * self.final_dim:] == i).sum().item())
            if counter == 0:
                weights.append(0)
            else:
                weights.append(1/counter)      
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
        if diff_metrix.shape[0] == 0:
            return torch.tensor([0]).cuda()
        loss = criterion(diff_metrix, new_label[self.final_dim * self.final_dim:])
        return loss

    def forward(self, x, labels):
        features = self.backbone.backbone(x)
        #features = self.projection(features.squeeze(-1))
        loss = self.info_nce_loss(features, labels)
        return loss, 0