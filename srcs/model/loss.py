# Passion4ever

import mindspore.nn as nn
from mindspore import ops


def pairwise_distance(x1, x2, p=2, eps=1e-6):
    euclidean_distance = ops.norm(x1 - x2 + eps, ord=p, dim=-1)
    return euclidean_distance


class ContrastiveLoss(nn.Cell):

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def construct(self, output1, output2, label):
        
        euclidean_distance = pairwise_distance(output1, output2)
        contrastive_loss = ops.mean((1-label) * ops.pow(euclidean_distance, 2) +     
                                      (label) * ops.pow(ops.clamp(self.margin - euclidean_distance, min=0.0), 2))  

        return contrastive_loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

