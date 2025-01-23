import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbones.vitst import (BitLinear, activation_quant_real,
                                    weight_quant_real)

if __name__ == "__main__":
    layer = nn.Linear(2, 4096)
    layer_bit = BitLinear(2, 4096)

    print(
        f"fp mean: {layer.weight.data.mean()}, max: {layer.weight.data.max()}, min: {layer.weight.data.min()}"
    )
    print(
        f"bit mean: {layer_bit.weight.data.mean()}, max: {layer_bit.weight.data.max()}, min: {layer_bit.weight.data.min()}"
    )
