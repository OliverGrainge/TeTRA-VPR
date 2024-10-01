import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


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


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(
    log_a, log_b, M, num_iters: int = 20, reg: float = 1.0
) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


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
        p = get_matching_probs(p, self.dust_bin, 3)
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

        return nn.functional.normalize(f, p=2, dim=-1)


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
        # ---- Train hyperparameters
        lr=0.03,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },
        # ----- Loss
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparametersge
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = (
            []
        )  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = DINOv2(model_name=backbone_arch, **backbone_config)
        self.aggregator = SALAD(**agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def DinoSalad(
    backbone: str = "dinov2_vitb14",
    pretrained=True,
    backbone_args=None,
    agg_args=None,
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
    model = VPRModel(
        backbone_arch=backbone,
        backbone_config=backbone_args,
        agg_arch="SALAD",
        agg_config=agg_args,
    )

    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
            map_location=torch.device("cpu"),
        )
    )
    return model
