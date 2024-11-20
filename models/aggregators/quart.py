import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


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
        B = x.shape[0]
        t = x[:, 0]
        f = x[:, 1:]
        patch_size = int((f.numel() / (B * self.num_channels)) ** 0.5)
        x = f.reshape((B, patch_size, patch_size, self.num_channels)).permute(
            0, 3, 1, 2
        )

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
        return f


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, layer_type=nn.Linear):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            layer_type(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            layer_type(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CLSAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, layer_type=nn.Linear):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = layer_type(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(layer_type(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Compute attention only for the cls token
        q_cls = q[:, :, 0:1, :]  # Query for cls token
        dots_cls = torch.matmul(q_cls, k.transpose(-1, -2)) * self.scale
        attn_cls = self.attend(dots_cls)
        attn_cls = self.dropout(attn_cls)

        return attn_cls


class Attention(nn.Module):
    def __init__(self, dim, dim_head=768, dropout=0.0, layer_type=nn.Linear):
        super().__init__()
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = layer_type(dim, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(layer_type(dim_head, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = qk  # No need to rearrange for single head
        q_cls = q[:, 0:1, :]
        dots = torch.matmul(q_cls, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        return attn[:, 0, 1:]


class ConvFeatureAggregator(nn.Module):
    def __init__(self, num_channels, local_feature_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels, local_feature_dim, kernel_size=kernel_size, stride=kernel_size
        )
        self.fc = nn.Linear(local_feature_dim, local_feature_dim)
        self.ln1 = nn.LayerNorm(local_feature_dim)
        self.ln2 = nn.LayerNorm(local_feature_dim)
        self.relu = nn.ReLU()
        self.local_feature_dim = local_feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), self.local_feature_dim, -1).permute(0, 2, 1)
        x = self.relu(self.ln1(x))
        x = self.ln2(self.fc(x))
        x = self.relu(x)
        return x


"""
class QuART(nn.Module): 
    def __init__(self, local_feature_dim=128, n_tokens=257,num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, dropout=0.3):
        super().__init__()
        self.salad = SALAD(num_channels, num_clusters, cluster_dim, token_dim, dropout)
        self.attn = Attention(dim=num_channels, dim_head=num_channels, dropout=0.0)
        self.H = int((n_tokens-1)**0.5)

        self.conv_feat4 = ConvFeatureAggregator(num_channels, local_feature_dim, self.H//2)
        self.conv_feat16 = ConvFeatureAggregator(num_channels, local_feature_dim, self.H//4)   
        self.cls_proj = nn.Sequential(nn.Linear(num_channels, local_feature_dim), nn.LayerNorm(local_feature_dim), nn.ReLU())
        self.local_feature_dim = local_feature_dim

    def forward(self, x): 
        B = x.shape[0]
        global_desc = self.salad(x)
        # extract attention map across local tokens
        attn = self.attn(x)
        # Extract features and reshape into image shape
        feats_img = x[:, 1:, :].permute(0, 2, 1).view(x.size(0), -1, int((x.size(1) - 1)**0.5), int((x.size(1) - 1)**0.5))
        # extract local features
        local_feats1 = self.cls_proj(x[:, 0, :]).unsqueeze(2).permute(0, 2, 1)
        local_feats4 = self.conv_feat4(feats_img)
        local_feats16 = self.conv_feat16(feats_img)
        # Stack and permute features
        local_feats = torch.cat([local_feats1, local_feats4, local_feats16], dim=1)
        local_feats = nn.functional.normalize(local_feats, p=2, dim=-1)
        attn_map = attn.view(attn.shape[0], self.H, self.H)
        attn1 = F.max_pool2d(attn_map, kernel_size=self.H).view(B, -1)
        attn4 = F.max_pool2d(attn_map, kernel_size=self.H//2).view(B, -1)
        attn16 = F.max_pool2d(attn_map, kernel_size=self.H//4).view(B, -1)
        local_attn = torch.cat([attn1, attn4, attn16], dim=1)
        return {"global_desc":global_desc, "local_desc":local_feats[:, :5, :], "local_attn":local_attn[:, :5]}
"""


class QuART(nn.Module):
    def __init__(
        self,
        local_feature_dim=64,
        n_tokens=257,
        num_channels=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        self.salad = SALAD(num_channels, num_clusters, cluster_dim, token_dim, dropout)
        self.fc = nn.Linear(num_channels, local_feature_dim)

    def forward(self, x):
        global_desc = self.salad(x)
        local = self.fc(x[:, 1:, :])
        local_desc = local
        return {"global_desc": global_desc, "local_desc": local_desc}
