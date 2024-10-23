import numpy as np
import torch
import torch.nn as nn
import torchvision



def ResNet(model_name="resnet50", pretrained=True, layers_to_freeze=2, layers_to_crop=[]):
    backbone = getattr(torchvision.models, model_name.lower())()
    cosplace = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=model_name, fc_output_dim=512)
    new_sd = {k1: v2 for (k1, v1), (k2, v2) in zip(backbone.state_dict().items(), cosplace.state_dict().items())
              if v1.shape == v2.shape}
    backbone.load_state_dict(new_sd, strict=False)

    for name, child in backbone.named_children():
        if name == "layer3":  # Freeze layers before conv_3
            break
        for params in child.parameters():
            params.requires_grad = False

    layers = list(backbone.children())[:-2] 
    backbone = torch.nn.Sequential(*layers)
    return backbone