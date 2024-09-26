import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import aggregators, backbones


def get_backbone(image_size, backbone_arch="resnet50", backbone_config={}):
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
            return backbones.ResNet(**backbone_config["resnet18"])
        elif "50" in backbone_arch.lower():
            return backbones.ResNet(**backbone_config["resnet50"])

    elif "efficient" in backbone_arch.lower():
        return backbones.EfficientNet(**backbone_config["efficientnet"])

    elif "swin" in backbone_arch.lower():
        return backbones.Swin(**backbone_config["swin"])

    elif "dinov2" in backbone_arch.lower():
        if torch.cuda.is_available():
            return backbones.DINOv2(**backbone_config["dinov2"]).cuda()
        else:
            raise Exception("Dinov2 not available without cuda")

    elif "vit" in backbone_arch.lower():
        if "ternary" in backbone_arch.lower():
            if "small" in backbone_arch.lower():
                return backbones.Ternary_ViT_Small()
            elif "base" in backbone_arch.lower():
                return backbones.Ternary_ViT_Base()
            elif "large" in backbone_arch.lower():
                return backbones.Ternary_ViT_Large()
        else:
            if "small" in backbone_arch.lower():
                return backbones.ViT_Small()
            elif "base" in backbone_arch.lower():
                return backbones.ViT_Base()
            elif "large" in backbone_arch.lower():
                return backbones.ViT_Large()

    elif "mobilevit" in backbone_arch.lower():
        backbone_arch["mobilevit"]["image_size"] = image_size
        if "ternary" in backbone_arch.lower():
            return backbones.Ternary_MobileViT(**backbone_config["mobilevit"])
        else:
            return backbones.MobileViT(**backbone_config["mobilevit"])

    elif "cct" in backbone_arch.lower():
        backbone_config["cct"]["image_size"] = image_size
        if "ternary" in backbone_arch.lower():
            return backbones.Ternary_CCT(**backbone_config["cct"])
        else:
            return backbones.CCT(**backbone_config["cct"])


def get_aggregator(agg_arch, agg_config, features_dim, image_size):
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
        agg_config["gem"]["in_dim"] = features_dim[0]
        return aggregators.GeM(**agg_config["gem"])

    elif "convap" in agg_arch.lower():
        agg_config["convap"]["in_channels"] = features_dim[0]
        return aggregators.ConvAP(**agg_config["convap"])

    elif "mixvpr" in agg_arch.lower():
        if "two_step" in agg_arch.lower():
            agg_config["mixvpr"]["in_channels"] = features_dim[0]
            agg_config["mixvpr"]["in_h"] = features_dim[1]
            agg_config["mixvpr"]["in_w"] = features_dim[2]
            return aggregators.MixVPR_TWO_STEP(agg_config["mixvpr"])

        if len(features_dim) == 3:
            agg_config["mixvpr"]["in_channels"] = features_dim[0]
            agg_config["mixvpr"]["in_h"] = features_dim[1]
            agg_config["mixvpr"]["in_w"] = features_dim[2]
        else:
            agg_config["mixvpr"]["channel_number"] = features_dim[1]
            agg_config["mixvpr"]["token_dim"] = features_dim[0]

        assert "out_channels" in agg_config["mixvpr"]
        assert "mix_depth" in agg_config["mixvpr"]
        return aggregators.MixVPR(features_dim, agg_config["mixvpr"])

    elif "salad" in agg_arch.lower():
        agg_config["salad"]["num_channels"] = features_dim[1]
        agg_config["salad"]["token_dim"] = features_dim[0]
        agg_config["salad"]["height"] = int(image_size[0])
        agg_config["salad"]["width"] = int(image_size[1])
        assert "num_clusters" in agg_config["salad"]
        assert "cluster_dim" in agg_config["salad"]
        return aggregators.SALAD(**agg_config["salad"])

    elif "cls" in agg_arch.lower():
        return aggregators.CLS()


class VPRModel(nn.Module):
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
        return x

def get_model(image_size, backbone_arch, agg_arch, model_config, normalize_output=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = get_backbone(image_size, backbone_arch, model_config["backbone_config"])
    backbone = backbone.to(device)
    image = torch.randn(3, *(image_size)).to(device)
    features = backbone(image[None, :])
    features_dim = list(features[0].shape)
    aggregation = get_aggregator(
        agg_arch, model_config["agg_config"], features_dim, image_size
    )
    aggregation = aggregation.to(device)
    model = VPRModel(backbone, aggregation, normalize=normalize_output)
    desc = aggregation(features)
    if type(desc) == tuple: 
        model.descriptor_dim = desc[0].shape[0]
    else: 
        model.descriptor_dim = desc.shape[1]
    return model
