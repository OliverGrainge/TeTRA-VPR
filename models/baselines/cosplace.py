import contextlib
import io

import torch


def CosPlaceR18D128(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR18D128 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=128,
        )
    model.name = f"CosPlaceR18D128"
    return model

def CosPlaceR18D32(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR18D32 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=32,
        )

    model.name = f"CosPlaceR18D32"
    return model

def CosPlaceR18D64(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR18D64 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=64,
        )
    model.name = f"CosPlaceR18D64"
    return model


def CosPlaceR50D128(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR50D128 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=128,
        )
    model.name = f"CosPlaceR50D128"
    return model

def CosPlaceR50D2048(noramlize=True):
    if not noramlize:
        raise Exception("CosPlace does not support noramlize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )
    model.name = f"CosPlaceR50D2048"
    return model

def CosPlaceR50D32(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR50D32 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=32,
        )
    model.name = f"CosPlaceR50D32"
    return model

def CosPlaceR50D64(normalize=True):
    if not normalize:
        raise Exception("CosPlaceR50D64 does not support normalize=False")
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=64,
        )
    model.name = f"CosPlaceR50D64"
    return model