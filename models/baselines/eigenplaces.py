import contextlib
import io
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.parameter import Parameter

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
}


AVAILABLE_TRAINED_MODELS = {
    # backbone : list of available fc_output_dim, which is equivalent to descriptors dimensionality
    "VGG16": [512],
    "ResNet18": [256, 512],
    "ResNet50": [128, 256, 512, 1024, 2048],
    "ResNet101": [128, 256, 512, 1024, 2048],
}


def gem(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class GeoLocalizationNet_(nn.Module):
    def __init__(self, backbone: str, fc_output_dim: int, normalize: bool = True):
        """Return a model_ for GeoLocalization.

        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert (
            backbone in CHANNELS_NUM_IN_LAST_CONV
        ), f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.normalize = normalize
        self.backbone, features_dim = _get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            # L2Norm() if normalize else nn.Identity()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        if self.normalize:
            x = F.normalize(x, p=2.0, dim=1)
        return x


def _get_torchvision_model(backbone_name: str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    return getattr(torchvision.models, backbone_name.lower())()


def _get_backbone(backbone_name: str) -> Tuple[torch.nn.Module, int]:
    backbone = _get_torchvision_model(backbone_name)

    logging.info("Loading pretrained backbone's weights from CosPlace")
    cosplace = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone=backbone_name,
        fc_output_dim=512,
    )
    new_sd = {
        k1: v2
        for (k1, v1), (k2, v2) in zip(
            backbone.state_dict().items(), cosplace.state_dict().items()
        )
        if v1.shape == v2.shape
    }
    backbone.load_state_dict(new_sd, strict=False)

    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones"
        )
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[
            :-2
        ]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim


def get_trained_model(
    backbone: str = "ResNet50", fc_output_dim: int = 2048, normalize: bool = True
) -> torch.nn.Module:
    """Return a model trained with EigenPlaces on San Francisco eXtra Large.

    Args:
        backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
        fc_output_dim (int): the output dimension of the last fc layer, equivalent to
            the descriptors dimension. Must be between 32 and 2048, depending on model's availability.

    Return:
        model (torch.nn.Module): a trained model.
    """
    print(
        f"Returning EigenPlaces model with backbone: {backbone} with features dimension {fc_output_dim}"
    )
    if backbone not in AVAILABLE_TRAINED_MODELS:
        raise ValueError(
            f"Parameter `backbone` is set to {backbone} but it must be one of {list(AVAILABLE_TRAINED_MODELS.keys())}"
        )
    try:
        fc_output_dim = int(fc_output_dim)
    except:
        raise ValueError(
            f"Parameter `fc_output_dim` must be an integer, but it is set to {fc_output_dim}"
        )
    if fc_output_dim not in AVAILABLE_TRAINED_MODELS[backbone]:
        raise ValueError(
            f"Parameter `fc_output_dim` is set to {fc_output_dim}, but for backbone {backbone} "
            f"it must be one of {list(AVAILABLE_TRAINED_MODELS[backbone])}"
        )
    model = GeoLocalizationNet_(backbone, fc_output_dim, normalize=normalize)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"https://github.com/gmberton/EigenPlaces/releases/download/v1.0/{backbone}_{fc_output_dim}_eigenplaces.pth",
            map_location=torch.device("cpu"),
        )
    )
    return model


def EigenPlacesR18D256(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_model(
            backbone="ResNet18", fc_output_dim=256, normalize=normalize
        )
        model.name = f"EigenPlacesR18D256"

    return model


def EigenPlacesR18D512(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_model(
            backbone="ResNet18", fc_output_dim=512, normalize=normalize
        )
        model.name = f"EigenPlacesR18D512"

    return model


def EigenPlacesR50D128(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_model(
            backbone="ResNet50", fc_output_dim=128, normalize=normalize
        )
        model.name = f"EigenPlacesR50D128"

    return model


def EigenPlacesR50D2048(normalize=True):

    model = get_trained_model(
        backbone="ResNet50", fc_output_dim=2048, normalize=normalize
    )
    model.name = f"EigenPlacesR50D2048"

    return model


def EigenPlacesR50D256(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_model(
            backbone="ResNet50", fc_output_dim=256, normalize=normalize
        )
        model.name = f"EigenPlacesR50D256"

    return model


def EigenPlacesR50D512(normalize=True):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = get_trained_model(
            backbone="ResNet50", fc_output_dim=512, normalize=normalize
        )
        model.name = f"EigenPlacesR50D512"

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


def QEigenPlacesR18D256(normalize=True):
    model = EigenPlacesR18D256()
    return quantize_convnet(model)


def QEigenPlacesR18D512(normalize=True):
    model = EigenPlacesR18D512()
    return quantize_convnet(model)


def QEigenPlacesR50D128(normalize=True):
    model = EigenPlacesR50D128()
    return quantize_convnet(model)


def QEigenPlacesR50D2048(normalize=True):
    model = EigenPlacesR50D2048()
    return quantize_convnet(model)


def QEigenPlacesR50D256(normalize=True):
    model = EigenPlacesR50D256()
    return quantize_convnet(model)


def QEigenPlacesR50D512(normalize=True):
    model = EigenPlacesR50D512()
    return quantize_convnet(model)
