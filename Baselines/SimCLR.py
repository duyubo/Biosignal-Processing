import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from Backbone.CNN_Backbone import *

class ResNetSimCLR(nn.Module):
    def __init__(self, backbone, args):
        super(ResNetSimCLR, self).__init__()
        self.n_views = args.n_views
        self.temperature = args.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.backbone = backbone
        self.projection = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
                
    def info_nce_loss(self, features, p_ids):
        labels = torch.cat([p_ids for i in range(self.n_views)], dim=0)
        # torch.arange(features.shape[0])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        #print('features: ', features.shape)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        #print('similarity matrix: ', similarity_matrix)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        #print('mask', mask)
        labels = labels[~mask].view(labels.shape[0], -1)
        #print('labels', labels)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        #print('similarity matrix: ', similarity_matrix)
        # select and combine multiple positives
        #print(similarity_matrix[labels.bool()])
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print('positives: ', positives)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        #print('negatives: ', negatives)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits / self.temperature

        return logits, labels

    def forward(self, x, p_ids):
        features = self.backbone(x)
        features = self.projection(features.squeeze(-1))
        logits, labels = self.info_nce_loss(features, p_ids)
        loss = self.criterion(logits, labels)
        return loss, 0