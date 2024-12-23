import importlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from . import aggregators, backbones

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NeuroCompress/"))
)

from NeuroPress.layers import LINEAR_LAYERS
from NeuroPress.models.base import Qmodel

LINEAR_REPR = [layer(12, 12).__repr__() for layer in LINEAR_LAYERS]


def find_best_match(target_string, list_of_strings):
    target_string = target_string.replace("vit", "")
    target_string = target_string.replace("small", "")
    target_string = target_string.replace("base", "")
    target_string = target_string.replace("large", "")
    target_string = target_string.replace("_", "")

    for s in list_of_strings:
        if s == target_string:
            return s  # Return the first match found
    return None



def get_backbone(backbone_arch, image_size):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where we sometimes need to crop the last residual block (ex. [4]). Defaults to [].

    Returns:
        model: the backbone as a nn.Model object
    """

    if "resnet" in backbone_arch.lower():
        if "18" in backbone_arch.lower():
            return backbones.ResNet(model_name="ResNet18")
        elif "50" in backbone_arch.lower():
            return backbones.ResNet(model_name="ResNet50")

    elif "vit" in backbone_arch.lower():
        match_layer_str = find_best_match(backbone_arch, LINEAR_REPR)
        if match_layer_str is None:
            if "small" in backbone_arch.lower():
                return backbones.ViT_Small(image_size=image_size)
            elif "base" in backbone_arch.lower():
                return backbones.ViT_Base(image_size=image_size)
            elif "large" in backbone_arch.lower():
                return backbones.ViT_Large(image_size=image_size)
            else:
                raise Exception("must choose small/medium/large")
        else:
            module = importlib.import_module(f"NeuroPress.layers.{match_layer_str}")
            layer_type = getattr(module, match_layer_str)
            if "small" in backbone_arch.lower():
                return backbones.QViT_Small(
                    image_size=image_size, layer_type=layer_type
                )
            elif "base" in backbone_arch.lower():
                return backbones.QViT_Base(image_size=image_size, layer_type=layer_type)
            elif "large" in backbone_arch.lower():
                return backbones.QViT_Large(
                    image_size=image_size, layer_type=layer_type
                )
            else:
                raise Exception("must choose small/medium/large")
    else:
        raise Exception("Backbone not available")


def get_aggregator(agg_arch, features_dim, image_size, out_dim=1000):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """

    if "gem" in agg_arch.lower():
        return aggregators.GeM(features_dim=features_dim, out_dim=2048)

    elif "convap" in agg_arch.lower():
        assert out_dim % 4 == 0
        return aggregators.ConvAP(s1=2, s2=2, out_channels=out_dim // 4)

    elif "mixvpr" in agg_arch.lower():
        config = {}
        config["in_channels"] = features_dim[1]
        config["in_h"] = int((features_dim[0] - 1) ** 0.5)
        config["in_w"] = int((features_dim[0] - 1) ** 0.5)
        config["out_channels"] = 1024
        config["mix_depth"] = 4
        config["mlp_ratio"] = 1
        config["out_rows"] = 4
        config["patch_size"] = 14
        config["image_size"] = image_size
        return aggregators.MixVPR(**config)

    elif "salad" in agg_arch.lower():
        config = {}
        config["num_channels"] = features_dim[1]
        config["token_dim"] = 256
        config["num_clusters"] = 64
        config["cluster_dim"] = 128
        return aggregators.SALAD(**config)

    elif "boq" in agg_arch.lower():
        return aggregators.BoQ(
            patch_size=14,
            image_size=image_size,
            in_channels=features_dim[1],
            proj_channels=512,
            num_queries=64,
            row_dim=12288 // 512,
        )


class VPRModel(Qmodel):
    def __init__(self, backbone, aggregation, normalize=True):
        super().__init__()
        self.backbone = backbone
        self.aggreagtion = aggregation
        self.normalize = normalize
        self.descriptor_dim = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggreagtion(x)
        if self.normalize == True:
            if isinstance(x, tuple):  # Check if x is a tuple
                x = list(x)  # Convert tuple to list to allow modification
                x[0] = F.normalize(x[0], p=2, dim=-1)
                x[1] = F.normalize(x[1], p=2, dim=-1)
                x = tuple(x)  # Optionally convert back to tuple if needed
            else:
                x = F.normalize(x, p=2, dim=-1)
        return {"global_desc": x}


def get_model(
    image_size=(224, 224),
    backbone_arch="vit_small",
    agg_arch="cls",
    out_dim=1024,
    normalize_output=True,
    preset=None,
):
    if preset is not None:
        if "EigenPlaces" in preset:
            module.importlib.import_module(f"models.presets.EigenPlaces")
        elif "CosPlaces" in preset:
            module = importlib.import_module(f"models.presets.CosPlaces")
        else:
            module = importlib.import_module(f"models.presets.{preset}")
        model = getattr(module, preset)
        return model()

    image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = get_backbone(backbone_arch, image_size=image_size)
    backbone = backbone.to(device)

    image = torch.randn(3, *(image_size)).to(device)
    features = backbone(image[None, :])
    features_dim = list(features[0].shape)
    aggregation = get_aggregator(agg_arch, features_dim, image_size, out_dim=out_dim)
    aggregation = aggregation.to(device)
    model = VPRModel(backbone, aggregation, normalize=normalize_output)
    desc = aggregation(features)
    # (model)
    if type(desc) == tuple:
        model.descriptor_dim = desc[0].shape[0]
    else:
        model.descriptor_dim = desc.shape[1]
    return model
