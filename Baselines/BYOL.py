import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from Backbone.CNN_Backbone import *

class BYOL_Backbone(nn.Module):
    def __init__(self, backbone, in_channels, mlp_hidden_size, projection_size):
        super(BYOL_Backbone, self).__init__()
        self.backbone = backbone
        self.projection = MLPHead(in_channels = in_channels,  mlp_hidden_size = mlp_hidden_size, projection_size = projection_size)

    def forward(self, x):
        x = self.backbone(x).squeeze(-1)
        x = self.projection(x)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone, args): 
        super(BYOL, self).__init__()
        self.online_network = BYOL_Backbone(backbone = backbone, mlp_hidden_size = args.mlp_hidden_size, in_channels = args.out_dim, projection_size = args.projection_size)
        self.target_network = BYOL_Backbone(backbone = backbone, mlp_hidden_size = args.mlp_hidden_size, in_channels = args.out_dim, projection_size = args.projection_size)
        self.predictor = MLPHead(in_channels = args.projection_size,  mlp_hidden_size = args.predictor_mlp_hidden_size, projection_size = args.projection_size)
        self.m = 0.996
        self.initializes_target_network()

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def forward(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1.detach())
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2.detach())
        return loss.mean(), 0

