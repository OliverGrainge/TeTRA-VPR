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
        x = x.permute(0, 2, 1)
        return (
            F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1)))
            .pow(1.0 / self.p)
            .unsqueeze(3)
        )


class GeM(nn.Module):
    """
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py

    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor
    """

    def __init__(self, features_dim, out_dim):
        super().__init__()
        self.gem = GeMPool()
        self.fc = nn.Linear(features_dim[1], out_dim)
        self.features_dim = features_dim

        self.name = f"GeM"

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=-1)
        return x
