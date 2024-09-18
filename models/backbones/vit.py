import torch
import torch.nn as nn
from transformers import ViTModel

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from einops import rearrange
from einops.layers.torch import Rearrange
import sys 





# Model definition (same as before)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTUntrained(nn.Module):
    def __init__(
        self,
        image_size=224,        # Smaller image size for reduced complexity
        patch_size=16,         # More patches for better granularity
        dim=384,               # Reduced embedding dimension
        depth=12,               # Fewer transformer layers
        heads=6,               # Fewer attention heads
        mlp_dim=1536,          # MLP layer dimension (4x dim)
        dropout=0.1,           # Regularization via dropout
        emb_dropout=0.1,       # Dropout for the embedding layer
        channels=3,            # RGB images
        dim_head=96           # Dimension of each attention head
    ):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
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
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x



class ViTPretrained(nn.Module):
    def __init__(
        self,
        image_size=[224 ,224],
        pretrained=True,
        layers_to_freeze=4,
        layers_to_truncate=12,
    ):
        super().__init__()
        if image_size[0] == 224:
            backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif image_size[1] == 384:
            backbone = ViTModel.from_pretrained("google/vit-base-patch16-384")
        else: 
            raise Exception("pretrained models only available with 224 or 384 image size")

        backbone.encoder.layer = backbone.encoder.layer[:layers_to_truncate]

        for p in backbone.parameters():
            p.requires_grad = False

        for name, child in backbone.encoder.layer.named_children():
            if int(name) > layers_to_freeze:
                for params in child.parameters():
                    params.requires_grad = True
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).last_hidden_state


def ViT(image_size=[224 ,224],
        pretrained=True,
        layers_to_freeze=4,
        layers_to_truncate=10):
    
    if pretrained: 
        return ViTPretrained(image_size=image_size, layers_to_freeze=layers_to_freeze, layers_to_truncate=layers_to_truncate)
    else: 
        return ViTUntrained(image_size=image_size[0], depth=layers_to_truncate)
    


def ViT_Base(image_size=[224 ,224],
        pretrained=False):
    
    if pretrained: 
        raise Exception("pretrain vision transformer is not available")
    else: 
        return ViTUntrained(image_size=image_size[0],
                            patch_size=16,
                            dim=768,
                            depth=12,
                            heads=12,
                            mlp_dim=3072,
                            dropout=0.1,
                            emb_dropout=0.1,
                            channels=3,
                            dim_head=64  # Usually dim_head = dim // heads
                        )


