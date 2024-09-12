import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch"""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class GeM(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeMPool()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
