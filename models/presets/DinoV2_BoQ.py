dependencies = ["torch", "torchvision"]

import contextlib
import io

import torch
import torch.nn as nn
import torchvision

# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------


class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
    ]

    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output

        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(
                f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14"
            )
            self.backbone_name = "dinov2_vitb14"

        self.dino = torch.hub.load("facebookresearch/dinov2", self.backbone_name)

        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False

        # unfreeze the last few blocks
        for block in self.dino.blocks[-unfreeze_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # remove the output norm layer of dino
        self.dino.norm = nn.Identity()  # remove the normalization layer

        self.out_channels = self.dino.embed_dim

    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches

    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks :]:
            x = blk(x)

        x = x[:, 1:]  # remove the [CLS] token

        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape  # or C = self.embed_dim
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x
    
    def forward_cls(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks :]:
            x = blk(x)

        x = x[:, 0]
        return x

    def state_dict(self, *args, **kwargs):
        sd = self.dino.state_dict(*args, **kwargs)
        new_sd = {}
        return sd


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # the following two lines are used during training.
        # for stability purposes
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(
        self,
        in_channels=1024,
        proj_channels=512,
        num_queries=32,
        num_layers=2,
        row_dim=32,
        normalize=True,
    ):
        super().__init__()
        self.normalize = normalize
        self.proj_c = torch.nn.Conv2d(
            in_channels, proj_channels, kernel_size=3, padding=1
        )
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList(
            [
                BoQBlock(in_dim, num_queries, nheads=in_dim // 64)
                for _ in range(num_layers)
            ]
        )

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns


class VPRModel(torch.nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x#, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}


def get_trained_boq(backbone_name="resnet50", output_dim=16384, normalize=True):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}"
        )
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(
            f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}"
        )

    backbone = DinoV2()
    # load the aggregator
    aggregator = BoQ(
        in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
        proj_channels=384,
        num_queries=64,
        num_layers=2,
        row_dim=output_dim // 384,  # 32 for dinov2
        normalize=normalize,
    )

    vpr_model = VPRModel(backbone=backbone, aggregator=aggregator)

    sd = torch.hub.load_state_dict_from_url(
        MODEL_URLS[f"{backbone_name}_{output_dim}"], map_location=torch.device("cpu")
    )

    # print(vpr_model.backbone.dino.norm.state_dict().keys())
    # for idx, key in enumerate(sd.keys()):
    # print(key, "------------------", list(vpr_model.state_dict().keys())[idx])
    # if key != list(vpr_model.state_dict().keys())[idx]:
    # print("============================================== key not found")
    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_URLS[f"{backbone_name}_{output_dim}"],
            map_location=torch.device("cpu"),
        ),
        strict=False,
    )

    vpr_model.name = f"DinoV2_BoQ"
    return vpr_model


def DinoV2_BoQ(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_boq(
            backbone_name="dinov2", output_dim=12288, normalize=normalize
        )
        model.name = f"DinoV2_BoQ"
    return model
