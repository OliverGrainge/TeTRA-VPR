# layer.py

#import bitblas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange


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
        print("=======================================", 1, self.out_features, self.in_features)
        """
        self.matmul_config = bitblas.MatmulConfig(
            M=1,
            N=out_features,
            K=in_features,
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
        """
        self.deployed = False

    def forward(self, x):
        if self.deployed:
            return self.deploy_forward(x)
        if self.training:
            return self.train_forward(x)
        else:
            return self.infer_forward(x)

    def train_forward(self, x):
        dqx = x + (activation_quant_fake(x)[0] - x).detach()
        dqw = self.weight + (weight_quant_fake(self.weight)[0] - self.weight).detach()
        out = F.linear(dqx, dqw)
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    def infer_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        if qx.dtype == torch.int8:
            out = torch.matmul(qx.to(x.dtype), self.qweight.T.to(x.dtype))
            out = torch.matmul(qx.float(), self.qweight.T.float())
        else:
            out = torch.matmul(qx, self.qweight.T)
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.inference_mode()
    def deploy_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        out = self.deploy_matmul(qx, self.qweight)
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
            #self.weight.data = self.weight.data.cpu()
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
            self.register_buffer("qweight", qweight.cuda())
            self.register_buffer("scale", scale.cuda()) 
        self = super().train(mode)
        return self

    def deploy(self):
        self.eval()
        print("--------------------------------", 1, self.out_features, self.in_features)
        #self.deploy_matmul = bitblas.Matmul(config=self.matmul_config)
        #self.qweight = self.deploy_matmul.transform_weight(self.qweight)
        del self.weight
        self.deployed = True

    def load_state_dict(self, state_dict, strict=True):
        if "qweight" in state_dict.keys():
            if (
                state_dict["qweight"].shape[0] == self.out_features
                and state_dict["qweight"].shape[1] < self.in_features
            ):
                print("loading in deployed mode")
                self.deploy()  # load the layer in deployed mode
            else:
                print("loading in eval mode")
                self.eval()  # load the
        else:
            print("loading in train mode")

        super(BitLinear, self).load_state_dict(state_dict, strict=strict)









class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            BitLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            BitLinear(hidden_dim, dim),
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
        self.to_qkv = BitLinear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(BitLinear(inner_dim, dim), nn.Dropout(dropout))

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
                            dim, mlp_dim, dropout=dropout
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

    def freeze_all_except_last_n(self, n):
        # Freeze patch embedding components
        for module in self.to_patch_embedding.modules():
            if isinstance(module, nn.Parameter):
                module.requires_grad = False

        # Freeze positional embeddings and cls token
        self.pos_embedding.requires_grad = False
        self.cls_token.requires_grad = False

        # Freeze transformer layers except last n
        n_layers = len(self.transformer.layers)
        layers_to_freeze = (
            self.transformer.layers[:-n] if n > 0 else self.transformer.layers
        )

        for idx, layer in enumerate(layers_to_freeze):
            for module in layer.modules():
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False

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


def ViT_Small(image_size=[224, 224]):
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
    )


def ViT_Base(image_size=[224, 224]):
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
    )


def ViT_Large(image_size=[224, 224]):
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
    )



if __name__ == "__main__":
    #bitblas.set_log_level("Info")
    model = ViT_Base(image_size=[224, 224])
    input = torch.randn(1, 3, 224, 224).cuda()
    model = model.cuda()
    model.train()
    out = model(input)
    model.deploy()
    out2 = model(input)
    print(out.flatten()[:5])
    print(out2.flatten()[:5])
