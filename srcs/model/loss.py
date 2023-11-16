# Passion4ever

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))  

        return contrastive_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)