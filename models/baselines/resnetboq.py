import contextlib
import io

import torch
import torch.nn as nn


class ResNetBoQ_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="resnet50",
            output_dim=16384,
        )

    def forward(self, x):
        return self.model(x)[0]


def ResNetBoQ(normalize=True):
    if not normalize:
        raise Exception("ResNet50_BoQ does not support normalize=False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = ResNetBoQ_model()
        model.name = f"ResNet50_BoQ"
    return model


class QConv(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        quant_mode="none",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_mode = quant_mode
        # Define quantization parameters for 8-bit
        self.num_bits = 8
        self.qmin = 0
        self.qmax = 2**self.num_bits - 1

    @classmethod
    def from_conv(cls, conv, quant_mode="none"):
        """
        Create a QConv layer from an existing nn.Conv2d layer.

        Args:
            conv (nn.Conv2d): The existing convolution layer
            quant_mode (str): Quantization mode to use

        Returns:
            QConv: A quantized version of the input convolution layer
        """
        qconv = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            quant_mode=quant_mode,
        )

        # Copy the weights and bias
        qconv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            qconv.bias.data.copy_(conv.bias.data)

        return qconv

    def fake_quantize(self, x, scale, zero_point):
        # Quantize: x_q = round(x / scale) + zero_point
        x_q = torch.round(x / scale) + zero_point
        # Clamp to ensure values are within range [qmin, qmax]
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        # Dequantize: x_dq = (x_q - zero_point) * scale
        x_dq = (x_q - zero_point) * scale
        return x_dq

    def forward(self, x):
        # Per-channel activation quantization
        x_abs_max = torch.max(torch.abs(x), dim=1, keepdim=True)[0].detach()
        act_scale = x_abs_max / (self.qmax / 2)
        act_zero_point = torch.full_like(act_scale, self.qmax / 2, dtype=torch.float32)
        x_q = self.fake_quantize(x, act_scale, act_zero_point)

        # Per-channel weight quantization
        w_abs_max = torch.max(torch.abs(self.weight), dim=1, keepdim=True)[0].detach()
        w_scale = w_abs_max / (self.qmax / 2)
        w_zero_point = torch.full_like(w_scale, self.qmax / 2, dtype=torch.float32)
        w_q = self.fake_quantize(self.weight, w_scale, w_zero_point)

        # Use quantized weights for convolution
        return nn.functional.conv2d(
            x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def quantize_convnet(model):
    """
    Recursively replace all nn.Conv2d layers in the model with QConv layers.

    Args:
        model (nn.Module): The model to quantize

    Returns:
        nn.Module: The quantized model
    """
    # Create a copy of the model to avoid modifying the original
    model = model.deepcopy() if hasattr(model, "deepcopy") else model

    # Recursively replace Conv2d modules
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Replace the Conv2d module with a QConv module
            setattr(model, name, QConv.from_conv(module, quant_mode="none"))
        elif len(list(module.children())) > 0:
            # If the module has children, recursively quantize them
            setattr(model, name, quantize_convnet(module))

    return model


def QResNetBoQ(normalize=True):
    model = ResNetBoQ(normalize)
    return quantize_convnet(model)
