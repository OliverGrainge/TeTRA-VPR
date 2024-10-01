import torch
import torch.nn as nn


class CLS(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim[1], out_dim)

    def forward(self, x):
        return self.fc(x[:, 0, :])
