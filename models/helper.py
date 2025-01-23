import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import aggregators, backbones


def get_backbone(backbone_arch, image_size):
    if "vitbaset" == backbone_arch.lower():
        return backbones.VitbaseT(image_size=image_size)
    elif "vitsmallt" == backbone_arch.lower():
        return backbones.VitsmallT(image_size=image_size)
    elif "vittinyt" == backbone_arch.lower():
        return backbones.VittinyT(image_size=image_size)
    else:
        raise Exception(f"Backbone {backbone_arch} not available")


def get_aggregator(agg_arch, features_dim, image_size, desc_divider_factor=None):
    config = {}
    if "gem" in agg_arch.lower():
        config["features_dim"] = features_dim
        config["out_dim"] = 512  # 2048
        if desc_divider_factor is not None:
            config["out_dim"] = config["out_dim"] // desc_divider_factor
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
        if desc_divider_factor is not None:
            config["out_channels"] = config["out_channels"] // desc_divider_factor
        return aggregators.MixVPR(**config)

    elif "salad" in agg_arch.lower():
        print("==========================================", features_dim[1])
        config["num_channels"] = features_dim[1]
        config["token_dim"] = 128  # 256
        config["num_clusters"] = 64
        config["cluster_dim"] = 64  # 128
        if desc_divider_factor is not None:
            config["cluster_dim"] = int(
                config["cluster_dim"] / np.sqrt(desc_divider_factor)
            )
            config["num_clusters"] = int(
                config["num_clusters"] / np.sqrt(desc_divider_factor)
            )

        return aggregators.SALAD(**config)

    elif "boq" in agg_arch.lower():
        config["patch_size"] = 16  # 14
        config["image_size"] = image_size
        config["in_channels"] = features_dim[1]
        config["proj_channels"] = 128  # 512
        config["num_queries"] = 32  # 64
        config["row_dim"] = (
            3072 // config["proj_channels"]
        )  # 12288 // config["proj_channels"]

        if desc_divider_factor is not None:
            config["proj_channels"] = config["proj_channels"] // desc_divider_factor
        return aggregators.BoQ(**config)


class VPRModel(nn.Module):
    def __init__(self, backbone, aggregation, normalize=True):
        super().__init__()
        self.backbone = backbone
        self.aggreagtion = aggregation
        self.normalize = normalize

        self.name = f"TeTRA-{self.backbone.name}_{self.aggreagtion.name}"

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggreagtion(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def deploy(self, use_bitblas=True):
        if hasattr(self.backbone, "deploy"):
            self.backbone.deploy(use_bitblas=use_bitblas)


def get_model(
    image_size=[224, 224],
    backbone_arch="vitsmall",
    agg_arch="salad",
    preset=None,
    desc_divider_factor=None,
    normalize=True,
):
    if preset is not None:
        module = importlib.import_module(f"models.presets.{preset}")
        model = getattr(module, preset)
        return model(normalize=normalize)

    image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    backbone = get_backbone(backbone_arch, image_size=image_size)
    image = torch.randn(3, *(image_size))
    features = backbone(image[None, :])
    features_dim = list(features[0].shape)
    aggregation = get_aggregator(
        agg_arch, features_dim, image_size, desc_divider_factor
    )
    model = VPRModel(backbone, aggregation, normalize=normalize)
    return model
