import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=512, device=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: Feature matrix with shape (batch_size, feat_dim).
            labels: True labels with shape (batch_size).
        """
        x = x.to(self.device)
        distmat = torch.cdist(x, self.centers)
        labels = labels.unsqueeze(1).to(self.device)
        distances = distmat.gather(1, labels)
        loss = distances.mean()
        return loss



class MLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None, device=None):        
        '''    
        These losses are described in the following papers: 
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        Initialize the margin loss function.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (classes).
            loss_type (str): Type of angular margin loss ('arcface', 'sphereface', 'cosface').
            eps (float): Small constant for numerical stability.
            s (float): Scaling factor.
            m (float): Margin.
            device (torch.device): Device for computation (CPU or GPU).
        '''
        super(MLoss, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        self.s = s if s else (64.0 if loss_type in ['arcface', 'sphereface'] else 30.0)
        self.m = m if m else (0.5 if loss_type == 'arcface' else (1.35 if loss_type == 'sphereface' else 0.37))
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False).to(self.device)
        self.eps = eps

    def forward(self, x, labels):
        '''
        Forward pass for the margin loss.

        Args:
            x (Tensor): Input tensor with shape (N, in_features).
            labels (Tensor): Ground truth labels.

        Returns:
            Tensor: Computed loss.
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

      
        x = F.normalize(x, p=2, dim=1)
        for W in self.fc.parameters():
            W.data = F.normalize(W.data, p=2, dim=1)

        
        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

 
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        denominator = torch.clamp(denominator, min=1e-10) 
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    

def convert_label_to_similarity(normed_feature: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)


    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

 


    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class CircleLossSoftplus(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossSoftplus, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sp, sn = convert_label_to_similarity(F.normalize(features), labels)
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss





class CircleLossExp(nn.Module):
    def __init__(self, scale: int, margin: float, similarity='cos'):
        super(CircleLossExp, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.unsqueeze(0).expand(m, m).eq(labels.unsqueeze(1).expand(m, m)).float()

        pos_mask = mask.triu(diagonal=1)
        neg_mask = mask.logical_not().triu(diagonal=1)

        if self.similarity == 'dot':
            self.similarity_matrix = torch.matmul(feats, feats.t())
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            self.similarity_matrix = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        self.pos_pair_ = self.similarity_matrix[pos_mask == 1]
        self.neg_pair_ = self.similarity_matrix[neg_mask == 1]

        num_pos = self.pos_pair_.size(0)
        num_neg = self.neg_pair_.size(0)

        if num_neg > num_pos:
            self.neg_pair_ = self.neg_pair_[:num_pos]
        elif num_pos > num_neg:
            self.pos_pair_ = self.pos_pair_[:num_neg]

        print(f"Balanced size of pos_pair_: {self.pos_pair_.size()}")
        print(f"Balanced size of neg_pair_: {self.neg_pair_.size()}")

        alpha_p = torch.relu(-self.pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(self.neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (self.pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (self.neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = ((output2 - output1).pow(2)).sum(1) 
        losses = 0.5 * (target.float() * distances + ((1 + (-1 * target)).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)))

        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, and a negative sample
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-9  

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        distance_positive = (anchor - positive).pow(2).sum(1) + self.epsilon  
        distance_negative = (anchor - negative).pow(2).sum(1) + self.epsilon

        losses = F.relu(distance_positive - distance_negative + self.margin)
        

        return losses.mean()
