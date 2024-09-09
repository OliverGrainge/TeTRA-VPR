import torch 
import torch.nn as nn
from transformers import ViTModel


class ViT(nn.Module):
    def __init__(self,
                 model_name='vit224',
                 pretrained=True,
                 layers_to_freeze=4,
                 layers_to_truncate=10
                 ):
        super().__init__()
        if '224' in model_name:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        elif '384' in model_name:
            backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')

        backbone.encoder.layer = backbone.encoder.layer[:layers_to_truncate]

        for p in backbone.parameters():
            p.requires_grad = False

        for name, child in backbone.encoder.layer.named_children():
            if int(name) > layers_to_freeze:
                for params in child.parameters():
                    params.requires_grad = True
        self.backbone = backbone 

    def forward(self, x):
        return self.backbone(x).last_hidden_state