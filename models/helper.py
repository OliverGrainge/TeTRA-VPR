import torch
import torch.nn as nn
import torch.nn.functional as F

from . import aggregators, backbones
from .baselines import ALL_BASELINES


def get_backbone(backbone_arch, image_size):
    if "ternaryvitbase" == backbone_arch.lower():
        return backbones.TernaryVitBase(image_size=image_size)
    elif "ternaryvitsmall" == backbone_arch.lower():
        return backbones.TernaryVitSmall(image_size=image_size)
    else:
        raise Exception(f"Backbone {backbone_arch} not available")


def get_aggregator(agg_arch, features_dim, image_size):
    arch = agg_arch.lower()

    if "gem" in arch:
        return aggregators.GeM(features_dim=features_dim, out_dim=2048)

    elif "mixvpr" in arch:
        h_w = int((features_dim[0] - 1) ** 0.5)
        return aggregators.MixVPR(
            in_channels=features_dim[1],
            in_h=h_w,
            in_w=h_w,
            out_channels=1024,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=4,
            patch_size=14,
            image_size=image_size,
        )

    elif "salad" in arch:
        return aggregators.SALAD(
            num_channels=features_dim[1],
            token_dim=256,
            num_clusters=64,
            cluster_dim=128,
        )

    elif "boq" in arch:
        return aggregators.BoQ(
            patch_size=14,
            image_size=image_size,
            in_channels=features_dim[1],
            proj_channels=512,
            num_queries=64,
            row_dim=12288 // 512,
        )

    else:
        raise ValueError(f"Unknown aggregator architecture: {agg_arch}")


class VPRModel(nn.Module):
    def __init__(self, backbone, aggregation, normalize=True):
        super().__init__()
        self.backbone = backbone
        self.aggregation = aggregation
        self.normalize = normalize

        self.name = f"{self.backbone.name}_{self.aggregation.name}"

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def deploy(self, use_bitblas=True):
        if hasattr(self.backbone, "deploy"):
            self.backbone.deploy(use_bitblas=use_bitblas)





def get_model(
    image_size=[322, 322],
    backbone_arch="ternaryvitbase",
    agg_arch="boq",
    normalize=True,
    baseline_name=None,
):  
    
   
    if baseline_name is not None: 
        assert baseline_name.lower() in ALL_BASELINES.keys(), f"Baseline {baseline_name} not available, choose from {ALL_BASELINES.keys()}"
        return ALL_BASELINES[baseline_name.lower()]()
    image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    backbone = get_backbone(backbone_arch, image_size=image_size)
    image = torch.randn(3, *(image_size))
    features = backbone(image[None, :])
    features_dim = list(features[0].shape)
    aggregation = get_aggregator(agg_arch, features_dim, image_size)
    model = VPRModel(backbone, aggregation, normalize=normalize)
    return model
