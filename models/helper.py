import importlib

import torch
import torch.nn as nn

from . import aggregators, backbones


def get_backbone(backbone_arch, image_size):
    if "vitsmall" == backbone_arch.lower():
        return backbones.Vitsmall(image_size=image_size)
    elif "vitbase" == backbone_arch.lower():
        return backbones.Vitbase(image_size=image_size)
    elif "vitsmallt" == backbone_arch.lower():
        return backbones.VitsmallT(image_size=image_size)
    elif "vitbaset" == backbone_arch.lower():
        return backbones.VitbaseT(image_size=image_size)
    elif "vitsmallnt" == backbone_arch.lower():
        return backbones.VitsmallNT(image_size=image_size)
    elif "vitbasent" == backbone_arch.lower():
        return backbones.VitbaseNT(image_size=image_size)
    else:
        raise Exception(f"Backbone {backbone_arch} not available")


def get_aggregator(agg_arch, features_dim, image_size):
    config = {}
    if "gem" in agg_arch.lower():
        config["features_dim"] = features_dim
        config["out_dim"] = 2048
        return aggregators.GeM(**config)

    elif "mixvpr" in agg_arch.lower():
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
        config["num_channels"] = features_dim[1]
        config["token_dim"] = 256
        config["num_clusters"] = 64
        config["cluster_dim"] = 128
        return aggregators.SALAD(**config)

    elif "boq" in agg_arch.lower():
        config["patch_size"] = 14
        config["image_size"] = image_size
        config["in_chanels"] = features_dim[1]
        config["proj_channels"] = 512
        config["num_queries"] = 64
        config["row_dim"] = 12288 // 512
        return aggregators.BoQ(**config)


class VPRModel(nn.Module):
    def __init__(self, backbone, aggregation):
        super().__init__()
        self.backbone = backbone
        self.aggreagtion = aggregation

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggreagtion(x)
        return x

    def deploy(self):
        if hasattr(self.backbone, "deploy"):
            self.backbone.deploy()

    def __str__(self):
        return f"{str(self.backbone)}_{str(self.aggreagtion)}"


def get_model(
    image_size=[224, 224],
    backbone_arch="vitsmall",
    agg_arch="salad",
    preset=None,
):
    if preset is not None:
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

    aggregation = get_aggregator(agg_arch, features_dim, image_size)
    aggregation = aggregation.to(device)
    model = VPRModel(backbone, aggregation)
    return model
