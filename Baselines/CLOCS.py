import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from itertools import combinations
from Backbone.CNN_Backbone import *

class CLOCS(nn.Module):
    def __init__(self, backbone, args):
        super(CLOCS, self).__init__()
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.backbone = backbone
        
    def info_nce_loss(self, features, pids):
        pids = pids[1]
        batch_size = features.shape[0]//self.n_views
        latent_embeddings = torch.stack([features[:batch_size, :], features[batch_size:, :]], dim = 0)
        latent_embeddings = latent_embeddings.permute(1, 2, 0)
        pids = np.array([str(p.item()) for p in pids],dtype=np.object)   
        pid1,pid2 = np.meshgrid(pids,pids)
        pid_matrix = pid1 + '-' + pid2
        pids_of_interest = np.unique(pids + '-' + pids) 
        bool_matrix_of_interest = np.zeros((len(pids),len(pids)))
        for pid in pids_of_interest:
            bool_matrix_of_interest += pid_matrix == pid

        rows1,cols1 = np.where(np.triu(bool_matrix_of_interest,1))
        rows2,cols2 = np.where(np.tril(bool_matrix_of_interest,-1))

        nviews = set(range(self.n_views))
        view_combinations = combinations(nviews,2)
    
        loss = 0
        ncombinations = 0
        for combination in view_combinations:
            view1_array = latent_embeddings[:,:,combination[0]]  
            view2_array = latent_embeddings[:,:,combination[1]] 
            norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
            norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
            sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
            norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
            argument = sim_matrix/(norm_matrix * self.temperature)
            sim_matrix_exp = torch.exp(argument)
        
            triu_elements = sim_matrix_exp[rows1,cols1]
            tril_elements = sim_matrix_exp[rows2,cols2]

            #print(rows1, cols1, rows2, cols2)
            diag_elements = torch.diag(sim_matrix_exp)
                
            triu_sum = torch.sum(sim_matrix_exp,1)
            tril_sum = torch.sum(sim_matrix_exp,0)
                
            loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
                
            loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
                
            loss = loss_diag1 + loss_diag2
            loss_terms = 2
            if len(rows1) > 0:
                loss += loss_triu 
                loss_terms += 1
                
            if len(rows2) > 0:
                loss += loss_tril 
                loss_terms += 1
            
            ncombinations += 1
        loss = loss/(loss_terms*ncombinations)
        return loss

    def forward(self, x, p_ids):
        features = self.backbone(x)
        features = features.squeeze(-1)
        loss = self.info_nce_loss(features, p_ids)
        return loss, 0