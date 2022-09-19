import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from Backbone.CNN_Backbone import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class WCL(nn.Module):
    def __init__(self, backbone, args):
        super(WCL, self).__init__()
        self.net = backbone
        self.head1 = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
        self.head2 = MLPHead(in_channels = args.out_dim,  mlp_hidden_size = args.mlp_hidden_size, projection_size = args.projection_size)
        self.temperature = args.temperature
        self.criteria =  nn.CrossEntropyLoss().cuda()
    
    @torch.no_grad()
    def build_connected_component(self, dist, subject_ids = None, epoch = 0):
        b = dist.size(0)
        dist = dist - torch.eye(b, b, device='cuda') * 2
        x = torch.arange(b, device='cuda').unsqueeze(1).repeat(1,1).flatten()
        y = torch.topk(dist, 1, dim=1, sorted=False)[1].flatten()
        rx = torch.cat([x, y]).cpu().numpy()
        ry = torch.cat([y, x]).cpu().numpy()
        v = np.ones(rx.shape[0])

        """Add intra-inter connected components"""
        if subject_ids.unique().shape[0] > 1 and epoch > 10:
            # mask the same subject pair to -1 distance
            m_x, m_y = torch.meshgrid(subject_ids, subject_ids)
            dist = dist - 2 * (m_x == m_y).cuda()
            # get the inter-subject closest data
            y1 = torch.topk(dist, 1, 
            
            1, sorted=False)[1].flatten()
            rx = torch.cat([x, y, x, y1]).cpu().numpy()
            ry = torch.cat([y, x, y1, x]).cpu().numpy()
            v = np.ones(rx.shape[0])
        """End"""   

        graph = csr_matrix((v, (rx, ry)), shape=(b,b))
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device='cuda')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T)
        return mask

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is not None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * diagnal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, x1, x2, p_labels = None, epoch = 0):
        t = self.temperature
        b = x1.size(0)
        bakcbone_feat1 = self.net(x1)
        bakcbone_feat2 = self.net(x2)
        bakcbone_feat1 = bakcbone_feat1.squeeze(-1)
        bakcbone_feat2 = bakcbone_feat2.squeeze(-1)
        feat1 = F.normalize(self.head1(bakcbone_feat1))
        feat2 = F.normalize(self.head1(bakcbone_feat2))
        prob = torch.cat([feat1, feat2]) @ torch.cat([feat1, feat2]).T / t
        diagnal_mask = (1 - torch.eye(prob.size(0), prob.size(1), device='cuda')).bool()
        logits = torch.masked_select(prob, diagnal_mask).reshape(prob.size(0), -1)

        first_half_label = torch.arange(b - 1, 2 * b - 1).long().cuda()
        second_half_label = torch.arange(0, b).long().cuda()
        labels = torch.cat([first_half_label, second_half_label])

        feat1 = F.normalize(self.head2(bakcbone_feat1))
        feat2 = F.normalize(self.head2(bakcbone_feat2))
        all_feat1 = feat1
        all_feat2 = feat2
        all_bs = all_feat1.size(0)
        distance_metrix1 = all_feat1 @ all_feat1.T
        distance_metrix2 = all_feat2 @ all_feat2.T
        mask1 = self.build_connected_component(distance_metrix1, p_labels[1], epoch = epoch).float()
        mask2 = self.build_connected_component(distance_metrix2, p_labels[1], epoch = epoch).float()

        """For testing the pseudo label, should be deleted during pretraining process"""
        m_x, m_y = torch.meshgrid(p_labels[0], p_labels[0])
        binary_metrix = (m_x == m_y)
        tp_metrix = torch.logical_and((mask1 == 1), binary_metrix.cuda()).sum()#dim = -1
        total_true = (mask1 == 1).sum()#dim = -1
        pseudo_acc = (tp_metrix/total_true)#.mean()

        m_x, m_y = torch.meshgrid(p_labels[0], p_labels[0])
        binary_metrix = (m_x == m_y)
        tp_metrix = torch.logical_and((mask2 == 1), binary_metrix.cuda()).sum()
        total_true = (mask2 == 1).sum()
        pseudo_acc += (tp_metrix/total_true)
        pseudo_acc /= 2
        """End"""

        """Testing the change of cluster center, should be deleted during pretraining process"""
        label_dict = p_labels[0].unique()
        p_dict = p_labels[1].unique()
        diversity_loss = 0
        count = 0
        for i in label_dict:
            all_feat1_i_mean = all_feat1[(p_labels[0] == i)].mean(dim = 0)
            all_feat1_i_mean = all_feat1_i_mean.reshape(all_feat1_i_mean.shape[0], 1)
            for p in p_dict:
                all_feat1_i_p_mean = all_feat1[torch.logical_and((p_labels[0] == i), (p_labels[1] == p))]
                if all_feat1_i_p_mean.shape[0] > 0:
                    all_feat1_i_p_mean = all_feat1_i_p_mean.mean(dim = 0)
                    diversity_loss += (all_feat1_i_p_mean @ all_feat1_i_mean).mean()
                    count += 1
        diversity_loss /= count
        """End"""

        diagnal_mask = torch.eye(all_bs, all_bs, device='cuda')
        graph_loss = 0 
        graph_loss =  self.sup_contra(feat1 @ all_feat1.T / t, mask2, diagnal_mask)
        graph_loss += self.sup_contra(feat2 @ all_feat2.T / t, mask1, diagnal_mask)
        graph_loss /= 2
        
        loss = graph_loss + self.criteria(logits, labels)
        return loss, logits, labels, pseudo_acc#, diversity_loss#
