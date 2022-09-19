import torch
import torch.nn as nn
from Backbone.CNN_Backbone import *

class Moco_Backbone(nn.Module):
    def __init__(self, backbone, in_channels, mlp_hidden_size, projection_size):
        super(Moco_Backbone, self).__init__()
        self.backbone = backbone
        self.projection = MLPHead(in_channels = in_channels,  mlp_hidden_size = mlp_hidden_size, projection_size = projection_size)

    def forward(self, x):
        x = self.backbone(x).squeeze(-1)
        x = self.projection(x)
        return x


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, backbone, args):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()
        self.T = args.temperature
        self.base_encoder = Moco_Backbone(backbone = backbone, mlp_hidden_size = args.mlp_hidden_size, in_channels = args.out_dim, projection_size = args.projection_size)
        self.momentum_encoder = Moco_Backbone(backbone = backbone, mlp_hidden_size = args.mlp_hidden_size, in_channels = args.out_dim, projection_size = args.projection_size)

        self.predictor = MLPHead(in_channels = args.projection_size,  mlp_hidden_size = args.predictor_mlp_hidden_size, projection_size = args.projection_size)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            if self.train:
                self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), 0


