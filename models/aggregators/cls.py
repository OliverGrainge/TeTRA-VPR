import torch
import torch.nn as nn


class CLS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 0, :]
