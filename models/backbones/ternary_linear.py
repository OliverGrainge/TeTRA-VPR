import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_ternary(tensor):
    assert tensor.dim() == 2, "Input must be a 2D tensor."

    allowed_values = torch.tensor([-1, 0, 1], device=tensor.device)
    if not torch.all(torch.isin(tensor, allowed_values)):
        raise ValueError("weight values must be only -1, 0, or 1")

    assert tensor.shape[1] % 4 == 0, "tensor.shape[1] must be divisible by 4"

    tensor += 1  # shift values to be 0, 1, 2

    # Flatten tensor and group into chunks of 4 values
    h, w = tensor.shape
    flat = tensor.flatten().view(-1, 4)

    # Pack 4 values into each byte
    packed = (flat[:, 0] << 6) | (flat[:, 1] << 4) | (flat[:, 2] << 2) | flat[:, 3]
    return packed.view(h, -1)


def unpack_ternary(packed):
    h, w = packed.shape
    w *= 4
    flat_packed = packed.flatten()

    # Extract 4 values per uint8
    unpacked = torch.stack(
        [
            (flat_packed >> 6) & 0b11,
            (flat_packed >> 4) & 0b11,
            (flat_packed >> 2) & 0b11,
            flat_packed & 0b11,
        ],
        dim=1,
    ).flatten()

    unpacked -= 1  # shift values back to -1, 0, 1
    unpacked = unpacked[: h * w]
    return unpacked.view(h, w)


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


class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(TernaryLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.deployed_real = False
        self.deployed_fake = False
        self.qfactor = 1

    def forward(self, x):
        if self.deployed_real:
            return self.deploy_forward_real(x)
        elif self.deployed_fake:
            return self.deploy_forward_fake(x)
        elif self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

    def train_forward(self, x):
        dqx = x + self.qfactor * (activation_quant_fake(x)[0] - x).detach()
        dqw = (
            self.weight
            + self.qfactor * (weight_quant_fake(self.weight)[0] - self.weight).detach()
        )
        dqy = F.linear(dqx, dqw, self.bias)
        return dqy

    @torch.no_grad()
    def eval_forward(self, x):
        device = x.device
        qx, act_scale = activation_quant_real(x)
        out = torch.matmul(qx.float(), self.qweight.T.float())
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def deploy_forward_real(self, x):
        # Quantize activation
        qx, act_scale = activation_quant_real(x)
        reshape_output = qx.ndim == 3
        if reshape_output:
            B, T, D = qx.shape
            qx = qx.reshape(-1, D)  # Flatten batch and time dimensions

        out = self.deploy_matmul.forward(qx, self.weight)
        if reshape_output:
            out = out.reshape(B, T, -1)
        out = out * (1.0 / (act_scale * self.scale))
        if self.bias is not None:
            out.add_(self.bias)

        return out

    @torch.no_grad()
    def deploy_forward_fake(self, x):
        qweight = unpack_ternary(self.weight)
        qx, act_scale = activation_quant_real(x)
        out = torch.matmul(qx.to(x.dtype), qweight.T.to(x.dtype))
        out = out / act_scale / self.scale
        if self.bias is not None:
            out += self.bias.to(out.dtype)
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        # Create new BitLinear instance with same dimensions as input linear layer
        bit_linear = cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None
        )

        # Copy parameters from linear layer
        bit_linear.weight = nn.Parameter(linear.weight.clone())
        if linear.bias is not None:
            bit_linear.bias = nn.Parameter(linear.bias.clone())
        else:
            bit_linear.bias = None

        # Initialize deployment flags
        bit_linear.deployed_real = False
        bit_linear.deployed_fake = False
        bit_linear.qfactor = 1

        return bit_linear

    def set_qfactor(self, qfactor):
        assert qfactor >= 0.0 and qfactor <= 1.0, "qfactor must be between 0.0 and 1.0"
        self.qfactor = qfactor

    def train(self, mode=True):
        if mode:
            self._buffers.clear()
        else:
            # Only quantize if we haven't deployed yet
            if not (self.deployed_real or self.deployed_fake):
                qweight, scale = weight_quant_real(self.weight)
                self.qweight = nn.Parameter(qweight, requires_grad=False)
                self.scale = scale
        self = super().train(mode)

    def deploy(self, use_bitblas=True, opt_M=None):
        try:
            import bitblas

            has_bitblas = True
        except ImportError:
            has_bitblas = False

        if has_bitblas and torch.cuda.is_available() and use_bitblas:
            # Real deployment with bitblas
            matmul_config = bitblas.MatmulConfig(
                M=[256, 512, 1024, 2048] if opt_M is None else opt_M,
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
            qweight, scale = weight_quant_real(self.weight)
            del self.weight
            if hasattr(self, "qweight"):
                del self.qweight
                del self.scale
            self.deploy_matmul = bitblas.Matmul(config=matmul_config)
            qweight = self.deploy_matmul.transform_weight(qweight)
            self.register_buffer("weight", qweight.cuda())
            self.register_buffer("scale", scale.cuda())
            if self.bias is not None:
                self.bias.data = self.bias.data.cuda()
            self.deployed_real = True
            self.deployed_fake = True
        else:
            # Fallback to fake deployment
            qweight, scale = weight_quant_real(self.weight)
            del self.weight
            if hasattr(self, "qweight"):
                del self.qweight
                del self.scale
            self.register_buffer("weight", pack_ternary(qweight))
            self.register_buffer("scale", scale.float())
            if self.bias is not None:
                self.bias.data = self.bias.data.float()
            self.deployed_fake = True
            self.deployed_real = False

    def state_dict(self, *args, **kwargs):
        has_qweight = False
        if hasattr(self, "qweight"):
            has_qweight = True
            qw = self.qweight
            s = self.scale
            del self.qweight
            del self.scale
        sd = super().state_dict(*args, **kwargs)
        if has_qweight:
            self.qweight = qw
            self.scale = s
        return sd
