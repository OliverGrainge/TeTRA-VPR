import contextlib
import io

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(
    Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([scores, bins], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
        normalize=True,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512), nn.ReLU(), nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

        self.normalize = normalize

    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x  # Extract features and token

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = log_optimal_transport(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        if self.normalize:
            f = nn.functional.normalize(f, p=2, dim=-1)
        return f


class ResNet(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[],
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        if "swsl" in model_name or "ssl" in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        else:
            if "resnext50" in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif "resnet50" in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif "101" in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif "152" in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif "34" in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif "18" in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif "wide_resnet50_2" in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError("Backbone architecture not recognized!")

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if "34" in model_name or "18" in model_name:
            out_channels = 512

        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        num_trainable_blocks=2,
        norm_layer=False,
        return_token=False,
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f"Unknown model name {model_name}"
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[: -self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks :]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f


def get_backbone(backbone_arch="resnet50", backbone_config={}):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if "resnet" in backbone_arch.lower():
        return ResNet(backbone_arch, **backbone_config)

    elif "dinov2" in backbone_arch.lower():
        return DINOv2(model_name=backbone_arch, **backbone_config)


def get_aggregator(agg_arch="ConvAP", agg_config={}, normalize=True):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """

    if "salad" in agg_arch.lower():
        assert "num_channels" in agg_config
        assert "num_clusters" in agg_config
        assert "cluster_dim" in agg_config
        assert "token_dim" in agg_config
        return SALAD(**agg_config, normalize=normalize)


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
        normalize=True,
    ):
        super().__init__()
        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = get_backbone(backbone_arch, backbone_config)
        self.aggregator = get_aggregator(agg_arch, agg_config, normalize=normalize)

        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def DinoSalad(
    backbone: str = "dinov2_vitb14",
    pretrained=True,
    backbone_args=None,
    agg_args=None,
    normalize=True,
) -> torch.nn.Module:
    """Return a DINOv2 SALAD model.

    Args:
        backbone (str): DINOv2 encoder to use ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14').
        pretrained (bool): If True, returns a model pre-trained on GSV-Cities (only available for 'dinov2_vitb14').
        backbone_args (dict): Arguments for the backbone (check models.backbones.dinov2).
        agg_args (dict): Arguments for the aggregation module (check models.aggregators.salad).
    Return:
        model (torch.nn.Module): the model.
    """
    assert (
        backbone in DINOV2_ARCHS.keys()
    ), f"Parameter `backbone` is set to {backbone} but it must be one of {list(DINOV2_ARCHS.keys())}"
    assert (
        not pretrained or backbone == "dinov2_vitb14"
    ), f"Parameter `pretrained` can only be set to True if backbone is 'dinov2_vitb14', but it is set to {backbone}"

    backbone_args = backbone_args or {
        "num_trainable_blocks": 4,
        "return_token": True,
        "norm_layer": True,
    }
    agg_args = agg_args or {
        "num_channels": DINOV2_ARCHS[backbone],
        "num_clusters": 64,
        "cluster_dim": 128,
        "token_dim": 256,
    }
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = VPRModel(
            backbone_arch=backbone,
            backbone_config=backbone_args,
            agg_arch="SALAD",
            agg_config=agg_args,
            normalize=normalize,
        )

        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                f"https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
                map_location=torch.device("cpu"),
            )
        )
    return model


model = DinoSalad(normalize=True)
img = torch.randn(1, 3, 322, 322)

out = model(img)
print(out[0].norm())


model = DinoSalad(normalize=False)
img = torch.randn(1, 3, 322, 322)

out = model(img)
print(out[0].norm())
