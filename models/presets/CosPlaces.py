import torch
import sys
import contextlib
import io


def CosPlaces():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model


def CosPlacesR50D128():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=128,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model



def CosPlacesR50D64():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=64,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model



def CosPlacesR50D32():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=32,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model



def CosPlacesR18D128():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=128,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model




def CosPlacesR18D64():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=64,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model



def CosPlacesR18D32():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=32,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model
