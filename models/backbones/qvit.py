import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import ViTModel

sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/../NeuroCompress"
    )
)

from NeuroPress.models.base import Qmodel


# Model definition (same as before)
class FeedForward(Qmodel):
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


class Attention(Qmodel):
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
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(Qmodel):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        attention_layer_type=nn.Linear,
        ff_layer_type=nn.Linear,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            layer_type=attention_layer_type,
                        ),
                        FeedForward(
                            dim, mlp_dim, dropout=dropout, layer_type=ff_layer_type
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(Qmodel):
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
        patch_layer_type=nn.Linear,
        attention_layer_type=nn.Linear,
        ff_layer_type=nn.Linear,
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
            patch_layer_type(patch_dim, dim),
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
            attention_layer_type=attention_layer_type,
            ff_layer_type=ff_layer_type,
        )

    def freeze_all_except_last_n(self, n):
        # Freeze patch embedding components
        for param in self.to_patch_embedding.parameters():
            param.requires_grad = False
            #if hasattr(module, 'freeze'):
                #module.freeze()
        
        # Freeze positional embeddings and cls token
        self.pos_embedding.requires_grad = False
        self.cls_token.requires_grad = False
        
        # Freeze transformer layers except last n
        n_layers = len(self.transformer.layers)
        layers_to_freeze = self.transformer.layers[:-n] if n > 0 else self.transformer.layers
        """
        for idx, layer in enumerate(layers_to_freeze):
            print(f"Freezing layer {idx}")
            for module in layer.modules():
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                #if hasattr(module, 'freeze'):
                #    module.freeze()
        """

        for idx, layer in enumerate(layers_to_freeze):
            print(f"Freezing layer {idx}")
            for param in layer.parameters():
                param.requires_grad = False

        for name, param in self.named_parameters(): 
            print(name, param.requires_grad)

        for module in self.modules():
            if hasattr(module, 'q_lambda'):
                module.q_labmda = 1.0

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


def QViT_Small(image_size=[224, 224], layer_type=nn.Linear):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,  # More patches for better granularity
        dim=384,  # Reduced embedding dimension
        depth=12,  # Fewer transformer layers
        heads=6,  # Fewer attention heads
        mlp_dim=1536,  # MLP layer dimension (4x dim)
        dropout=0.1,  # Regularization via dropout
        emb_dropout=0.1,  # Dropout for the embedding layer
        channels=3,  # RGB images
        dim_head=96,  # Dimension of each attention head
        patch_layer_type=nn.Linear,
        attention_layer_type=layer_type,
        ff_layer_type=layer_type,
    )


def QViT_Base(image_size=[224, 224], layer_type=nn.Linear):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        channels=3,
        dim_head=64,  # Usually dim_head = dim // heads
        patch_layer_type=nn.Linear,
        attention_layer_type=layer_type,
        ff_layer_type=layer_type,
    )


def QViT_Large(image_size=[224, 224], layer_type=nn.Linear):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=16,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0.1,
        emb_dropout=0.1,
        channels=3,
        dim_head=64,  # Usually dim_head = dim // heads
        patch_layer_type=nn.Linear,
        attention_layer_type=layer_type,
        ff_layer_type=layer_type,
    )
