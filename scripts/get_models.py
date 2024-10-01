import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch

from models.helper import get_model

model = get_model(preset="DinoSalad")


img = torch.randn(1, 3, 224, 224)
out = model(img)
print(out.shape)
