import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange


class Normalize(nn.Module):
    def __init__(self, eps=1e-05):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.mean(dim=-1, keepdim=True)) / (
            x.var(dim=-1, keepdim=True) + self.eps
        ) ** 0.5


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias


@torch.no_grad()
def activation_quant_fake(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    dqx = (x * scale).round().clamp_(-128, 127) / scale
    return dqx, scale


@torch.no_grad()
def activation_quant_real(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    qx = (x * scale).round().clamp_(-128, 127).type(torch.int8)
    return qx, scale


@torch.no_grad()
def weight_quant_fake(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    dqw = (w * scale).round().clamp_(-1, 1) / scale
    return dqw, scale


@torch.no_grad()
def weight_quant_real(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    qw = (w * scale).round().clamp_(-1, 1).type(torch.int8)
    return qw, scale


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert torch.cuda.is_available(), "CUDA is not available"

        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda())

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).cuda())
        else:
            self.register_parameter("bias", None)

        self.deployed = False

    def forward(self, x):
        if self.deployed:
            return self.deploy_forward(x)
        elif self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

    def train_forward(self, x):
        dqx = x + (activation_quant_fake(x)[0] - x).detach()
        dqw = self.weight + (weight_quant_fake(self.weight)[0] - self.weight).detach()
        out = F.linear(dqx, dqw)
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def eval_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        out = torch.matmul(qx.to(x.dtype), self.qweight.T.to(x.dtype))
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def deploy_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        if qx.ndim == 3:
            B, T, D = qx.shape
            qx = qx.view(B * T, D)
            reshape_output = True
        else:
            reshape_output = False
        out = self.deploy_matmul(qx, self.qweight)
        if reshape_output:
            out = out.view(B, T, -1)
        out = out / act_scale / self.scale

        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    def train(self, mode=True):
        if mode:
            self._buffers.clear()
            self.weight.data = self.weight.data.cuda()
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
        else:
            qweight, scale = weight_quant_real(self.weight)
            self.weight.data = self.weight.data.cpu()
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
            self.register_buffer("qweight", qweight.cuda())
            self.register_buffer("scale", scale.cuda())
        self = super().train(mode)
        return self

    def deploy(self):
        self.eval()
        try:
            import bitblas

            matmul_config = bitblas.MatmulConfig(
                M=[256, 512, 1024, 2048],
                N=self.out_features,
                K=self.in_features,
                A_dtype="int8",
                W_dtype="int2",
                accum_dtype="int32",
                out_dtype="int32",
                layout="nt",
                with_bias=False,
                group_size=None,
                with_scaling=False,
                with_zeros=False,
                zeros_mode=None,
            )
            self.deploy_matmul = bitblas.Matmul(config=matmul_config)
            self.qweight = self.deploy_matmul.transform_weight(self.qweight)
            self.deployed = True

        except ImportError as e:
            import logging

            logging.warning(
                "bitblas not installed, deploying in evaluation mode: %s", e
            )
        except Exception as e:
            import logging

            logging.error("An error occurred during deployment: %s", e)
        finally:
            del self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(dim),
            BitLinear(dim, hidden_dim),
            LayerScale(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Normalize(hidden_dim),
            BitLinear(hidden_dim, dim),
            LayerScale(dim),
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

        # Normalization layers
        self.norm1 = Normalize(dim)
        self.norm2 = Normalize(inner_dim)

        # Layer scaling
        self.layerscale1 = LayerScale(inner_dim * 3)
        self.layerscale2 = LayerScale(dim)

        # Attention mechanism
        self.to_qkv = BitLinear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Output transformation
        self.to_out = nn.Sequential(BitLinear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # compute q, k, v
        x = self.norm1(x)
        qkv = self.to_qkv(x)
        qkv = self.layerscale1(qkv)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # out projection
        out = self.norm2(out)
        out = self.to_out(out)
        out = self.layerscale2(out)
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
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(
                            dim,
                            mlp_dim,
                            dropout=dropout,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


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

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x

    def deploy(self):
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.deploy()

    def __str__(self):
        model_type = (
            "VitsmallST"
            if self.dim == 384
            else "VitbaseST" if self.dim == 768 else "Vit"
        )
        return f"{model_type}{self.image_size}"


def VitsmallST(image_size=[224, 224]):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,  # More patches for better granularity
        dim=384,  # Reduced embedding dimension
        depth=1,  # Fewer transformer layers 12
        heads=6,  # Fewer attention heads
        mlp_dim=1536,  # MLP layer dimension (4x dim)
        dropout=0.1,  # Regularization via dropout
        emb_dropout=0.1,  # Dropout for the embedding layer
        channels=3,  # RGB images
        dim_head=98,  # Dimension of each attention head (use a slightly larger value so down projection is suitable for bitblas kernel)
    )


def VitbaseST(image_size=[224, 224]):
    return ViT(
        image_size=image_size[0],  # Smaller image size for reduced complexity
        patch_size=14,
        dim=768,
        depth=1,  # 12
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        channels=3,
        dim_head=64,  # Usually dim_head = dim // heads
    )
