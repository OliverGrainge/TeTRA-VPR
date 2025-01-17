import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange


from models.helper import get_model


sd = torch.load("checkpoints/TeTRA-finetune/VitbaseT224_GeM-DescDividerFactor[2]/epoch=3-MSLS_val_q_R1=65.27.ckpt", weights_only=True)
model = get_model(backbone_arch="vitbaset", agg_arch="gem", image_size=(224, 224), desc_divider_factor=2)
model.eval()


sd = sd["state_dict"]


for key in sd.keys():
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(sd[key])
    else:
        print(f"Key {key} not found in state dict")
