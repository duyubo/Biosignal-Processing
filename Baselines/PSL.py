import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from Backbone.CNN_Backbone import *

"""
#PSL version 1
supervised version
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
"""
#PSL version 2
"""class PSL(nn.Module):
    def __init__(self, backbone, args):
        super(PSL, self).__init__()
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.backbone = backbone
        self.projection = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
        self.diff =  nn.MultiheadAttention(args.projection_size, 16, dropout=0.1)
        self.header = nn.Sequential(  nn.Linear(args.projection_size, args.projection_size),
                                      nn.ReLU(),
                                      nn.Linear(args.projection_size, args.final_dim * (args.final_dim -1) // 2 + 1), 
                                      )
        x_c, y_c = torch.meshgrid(torch.arange(args.final_dim), torch.arange(args.final_dim))
        self.map_indexes = torch.stack([x_c.triu(diagonal = 1), y_c.triu(diagonal = 1)]).permute(1, 2, 0).cuda()
        #self.average = torch.ones((args.projection_size, args.final_dim * (args.final_dim -1) // 2 + 1))
        self.final_dim = args.final_dim
        self.top_number = args.topk
        self.t = args.temperature
    def info_nce_loss(self, features, labels, true_labels):
        print(features.shape)
        #labels[0]: label, labels[1]: subject id
        batch_size = labels[0].shape[0]
        seq_l = features.shape[2]
        p_x, p_y = torch.meshgrid(labels[1], labels[1])
        p_binary_metrix = (p_x == p_y).cuda()

        l_x, l_y = torch.meshgrid(labels[0], labels[0])
        l_metrix = torch.stack([l_x, l_y]).permute(1, 2, 0).cuda()

        l_pseudo_metrix_or = torch.logical_or(l_x == -2, l_y == -2).cuda()
        l_pseudo_metrix_and = torch.logical_and(l_x == -2, l_y == -2).cuda()
        
        equal_label_metrix = torch.logical_and((l_x == l_y).cuda(), ~l_pseudo_metrix_or)
        i_x, i_y = torch.meshgrid(torch.arange(batch_size).cuda(), torch.arange(batch_size).cuda())
        i_metrix = torch.stack([i_x, i_y]).permute(1, 2, 0).cuda()
        
        l_metrix[equal_label_metrix] = 0

        #Supervised Loss for True Labels
        #mask for all classes pair from same subjects (with true label)
        binary_metrix = p_binary_metrix * ~l_pseudo_metrix_or * (i_x != i_y)
        labels_difference = l_metrix[binary_metrix]
        select_indexes = i_metrix[binary_metrix]
        batch_size = select_indexes.shape[0]
        true_label_features = features.index_select(dim = 0, index = select_indexes.reshape(-1))#
        true_label_features = true_label_features.reshape(batch_size, 2, true_label_features.shape[1], true_label_features.shape[2]).permute(1, 0, 3, 2)
        #features.shape = 2, batch_size, seq_length, hidden_dim
        
        diff_metrix, _ = self.diff(query = true_label_features[0], key = true_label_features[1], value = true_label_features[1], need_weights=False)
        diff_metrix = self.backbone.final(diff_metrix.transpose(1, 2))
        diff_metrix = diff_metrix.squeeze(-1)

        # get class pair (class i and class j, i!= j) from same subject
        unique_pair, new_label_true = torch.unique(torch.cat([self.map_indexes.reshape(-1, 2), labels_difference.sort()[0]]), dim = 0, return_inverse = True)
        final_metrix = self.header(diff_metrix)
        
        #Supervised Loss for Pseudo Labels
        #mask for all classes pair from same subjects (will have pseudo labels generated from true labels)
        binary_metrix = torch.logical_and(p_binary_metrix, l_pseudo_metrix_or)
        # make sure there are unlabeled data in a batch
        if binary_metrix.sum() > 0 and self.training:
            labels_difference = l_metrix[binary_metrix]
            select_indexes = i_metrix[binary_metrix]
            batch_size = select_indexes.shape[0]
            pseudo_label_features = features.index_select(dim = 0, index = select_indexes.reshape(-1))#
            pseudo_label_features = pseudo_label_features.reshape(batch_size, 2, pseudo_label_features.shape[1], pseudo_label_features.shape[2]).permute(1, 0, 3, 2)
            #features.shape = 2, batch_size, seq_length, hidden_dim
            #test pseudo label accuracy
            t_x, t_y = torch.meshgrid(true_labels, true_labels)
            t_metrix = torch.stack([t_x, t_y]).permute(1, 2, 0).cuda()
            t_metrix[t_x == t_y] = 0
            unique_pair_test, new_label_test = torch.unique(torch.cat([self.map_indexes.reshape(-1, 2), t_metrix[binary_metrix].sort()[0]]), dim = 0, return_inverse = True)
            
            diff_metrix_pseudo, _ = self.diff(query = pseudo_label_features[0], key = pseudo_label_features[1], value = pseudo_label_features[1], need_weights=False)
            diff_metrix_pseudo = self.backbone.final(diff_metrix_pseudo.transpose(1, 2))
            diff_metrix_pseudo = diff_metrix_pseudo.squeeze(-1)
            
            #assign pseudo label
            diff_metrix_pseudo1 = F.softmax(diff_metrix_pseudo/self.t, dim = 1)
            diff_metrix1 = F.softmax(diff_metrix/self.t, dim = 1)
            sim_metrix = diff_metrix_pseudo1 @ torch.log(diff_metrix1).T

            pseudo_labels = new_label_true[torch.topk(sim_metrix, self.top_number)[1]]
            high_confidence_pair = ((pseudo_labels[:, 0].unsqueeze(-1).repeat(1, self.top_number) == pseudo_labels).sum(dim = 1) == self.top_number) * (pseudo_labels[:, 0] != 0)
            diff_metrix_pseudo = diff_metrix_pseudo[high_confidence_pair]
            new_label_pseudo = pseudo_labels[:, 0][high_confidence_pair]
            new_label_test = new_label_test[self.final_dim * self.final_dim:][high_confidence_pair]
            
            if new_label_test.shape[0] != 0:
                pseudo_loss = (new_label_test == new_label_pseudo).sum()/new_label_test.shape[0]
            else:
                pseudo_loss = torch.tensor([0]).cuda()
            final_metrix_pseudo = self.header(diff_metrix_pseudo)
            #final_metrix = torch.cat([final_metrix, final_metrix_pseudo])
            #new_label_true = torch.cat([new_label_true, new_label_pseudo])
            pseudo_loss = torch.tensor([0]).cuda()
        else:
            pseudo_loss = torch.tensor([0]).cuda()
        weights = []
        counts = []
        check_metrix = (final_metrix.max(dim = 1)[1] == new_label_true[self.final_dim * self.final_dim:])
        accuracy = 0
        for i in range(unique_pair.shape[0]):
            i_mask = (new_label_true[self.final_dim * self.final_dim:] == i)
            counter = (i_mask.sum().item())
            if counter == 0:
                weight = 0
                weights.append(0)
            else:
                weight = 1/counter
                accuracy += (check_metrix[i_mask].sum()/i_mask.sum()) * (counter/new_label_true.shape[0])
                weights.append(weight)   
            counts.append(counter)   
        criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(weights)).cuda()
        if final_metrix.shape[0] == 0:
            return torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        loss = criterion(final_metrix, new_label_true[self.final_dim * self.final_dim:])
        return loss, accuracy

    def forward(self, x, labels, true_labels = None):
        features = self.backbone.backbone(x)
        #features = self.projection(features.squeeze(-1))
        loss, pseudo_loss = self.info_nce_loss(features, labels, true_labels)
        
        return loss, pseudo_loss"""

class PSL(nn.Module):
    def __init__(self, backbone, args):
        super(PSL, self).__init__()
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.backbone = backbone
        self.projection = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
        self.diff =  nn.MultiheadAttention(args.projection_size, 16, dropout=0.1)
        self.header = nn.Sequential(  nn.Linear(args.projection_size * 2, args.projection_size),
                                      nn.ReLU(),
                                      nn.Linear(args.projection_size, args.final_dim * (args.final_dim -1) // 2 + 1), #
                                      )
        
        #self.average = torch.ones((args.projection_size, args.final_dim * (args.final_dim -1) // 2 + 1))
        self.final_dim = args.final_dim * (args.final_dim -1) // 2 + 1
        self.top_number = args.topk
        self.t = args.temperature
    def info_nce_loss(self, features, labels, true_labels):
        #labels[0]: label, labels[1]: subject id
        batch_size = features.shape[0] // 2
        diff_metrix1, _ = self.diff(query = features[:batch_size], key = features[batch_size:], value = features[batch_size:], need_weights=False)
        diff_metrix2, _ = self.diff(query = features[batch_size:], key = features[:batch_size], value = features[:batch_size], need_weights=False)
        
        diff_metrix1 = self.backbone.final(diff_metrix1.transpose(1, 2))
        diff_metrix1 = diff_metrix1.squeeze(-1)

        diff_metrix2 = self.backbone.final(diff_metrix2.transpose(1, 2))
        diff_metrix2 = diff_metrix2.squeeze(-1)

        diff_metrix = torch.cat([diff_metrix1, diff_metrix2], dim = -1)
        final_metrix = self.header(diff_metrix)
        weights = []
        for i in range(self.final_dim):
            count = (labels[0] == i).sum().item()
            if count > 0:
                weights.append(1/count)
            else:
                weights.append(0)
        criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(weights)).cuda()
        loss = criterion(final_metrix, labels[0].cuda())
        accuracy = (final_metrix.max(dim = 1)[1] == labels[0].cuda()).sum()/labels[0].cuda().shape[0]
        return loss, accuracy
       

    def forward(self, x, labels, true_labels = None):
        features = self.backbone.backbone(x)
        features = features.permute(0, 2, 1)
        #features = self.projection(features)
        loss, pseudo_loss = self.info_nce_loss(features, labels, true_labels)
        
        return loss, pseudo_loss