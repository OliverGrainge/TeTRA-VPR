import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ImagenetViTWithCLS(nn.Module):
    def __init__(self, num_unfrozen_blocks=5):
        super().__init__()
        # Load the pre-trained ViT-B/16 model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze the last few transformer blocks
        for i in range(
            len(self.vit.encoder.layers) - num_unfrozen_blocks,
            len(self.vit.encoder.layers),
        ):
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        # Get the feature representations from the ViT model
        cls_token = self.vit(x)

        # L2 normalize the CLS token
        cls_token_normalized = nn.functional.normalize(cls_token, p=2, dim=1)
        return {"global_desc": cls_token_normalized}


def ImageNetViT(num_unfrozen_blocks=5):
    return ImagenetViTWithCLS(num_unfrozen_blocks=num_unfrozen_blocks)
