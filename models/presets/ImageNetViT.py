import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ImagenetViTWithCLS(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained ViT-B/16 model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove the classification head
        self.vit.heads = nn.Identity()
        
    def forward(self, x):
        # Get the feature representations from the ViT model
        cls_token = self.vit(x)
        
        # L2 normalize the CLS token
        cls_token_normalized = nn.functional.normalize(cls_token, p=2, dim=1)
        return {"global_desc": cls_token_normalized}

def ImageNetViT():
    return ImagenetViTWithCLS()
