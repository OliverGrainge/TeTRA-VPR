import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange

from models.backbones.ternary_linear import TernaryLinear


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, layer_type="TernaryLinear"):
        super().__init__()
        if layer_type.lower() == "ternarylinear":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                TernaryLinear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                TernaryLinear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        elif layer_type.lower() == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, layer_type="TernaryLinear"
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        # Normalization layers
        self.lnorm1 = nn.LayerNorm(dim)
        self.lnorm2 = nn.LayerNorm(inner_dim)

        # Attention mechanism
        if layer_type.lower() == "ternarylinear":
            self.to_qkv = TernaryLinear(dim, inner_dim * 3, bias=False)
        elif layer_type.lower() == "linear":
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Output transformation
        if layer_type.lower() == "ternarylinear":
            self.to_out = nn.Sequential(
                TernaryLinear(inner_dim, dim), nn.Dropout(dropout)
            )
        elif layer_type.lower() == "linear":
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, x, return_attn=False):
        # compute q, k, v
        x = self.lnorm1(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # attention
        attn_map = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn_map)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # out projection
        out = self.lnorm2(out)
        out = self.to_out(out)
        if return_attn:
            return out, attn_map
        else:
            return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            layer_type = "Linear" if i == depth - 1 else "ternarylinear"
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            layer_type=layer_type,
                        ),
                        FeedForward(
                            dim, mlp_dim, dropout=dropout, layer_type=layer_type
                        ),
                    ]
                )
            )

    def forward(self, x, return_attn=False):
        if return_attn:
            attentions = []
            for attn, ff in self.layers:
                x_attn, attn_map = attn(x, return_attn=True)
                attentions.append(attn_map)
                x = x_attn + x
                x = ff(x) + x
            return x, torch.stack(attentions).permute(1, 0, 2, 3, 4)
        else:
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,  # Smaller image size for reduced complexity
        patch_size=16,  # More patches for better granularity
        dim=384,  # Reduced embedding dimension
        depth=12,  # Fewer transformer layers
        heads=6,  # Fewer attention heads
        mlp_dim=1536,  # MLP layer dimension (4x dim)
        dropout=0.1,  # Regularization via dropout
        emb_dropout=0.1,  # Dropout for the embedding layer
        channels=3,  # RGB images
        dim_head=96,  # Dimension of each attention head
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.image_size = image_size
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )
        model_type = (
            "VittinyT"
            if self.dim == 192
            else "VitsmallT" if self.dim == 384 else "VitbaseT"
        )
        self.name = f"TeTRA-{model_type}{self.image_size}"

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x

    def forward_distill(self, x, return_attn=False):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, return_attn=return_attn)
        return x

    def deploy(self, use_bitblas=True):
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.deploy(use_bitblas=use_bitblas, opt_M=[512, 1024])

    def set_qfactor(self, qfactor):
        for module in self.modules():
            if isinstance(module, TernaryLinear):
                module.set_qfactor(qfactor)


def TernaryVitSmall(image_size=[224, 224]):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,  # More patches for better granularity
        dim=384,  # Reduced embedding dimension
        depth=12,  # 12,
        heads=6,  # Fewer attention heads
        mlp_dim=1536,  # MLP layer dimension (4x dim)
        dropout=0.05,  # Regularization via dropout
        emb_dropout=0.1,  # Dropout for the embedding layer
        channels=3,  # RGB images
        dim_head=96,  # Dimension of each attention head (use a slightly larger value so down projection is suitable for bitblas kernel)
    )


def TernaryVitBase(image_size=[224, 224]):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,
        dim=768,
        depth=12,  # 12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.05,
        channels=3,
        dim_head=64,  # Usually dim_head = dim // heads
    )
