import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(self, in_channels, spatial_dims, out_dim):
        super(FullyConnected, self).__init__()
        self.linear = nn.Linear(
            in_channels * spatial_dims[0] * spatial_dims[1], out_dim
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = x.flatten(1)
        return self.ln(self.linear(x))
