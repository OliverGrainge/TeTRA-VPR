import contextlib
import io

import torch


def EigenPlacesR50D512(normalize=True):
    if not normalize:
        raise Exception("EigenPlacesR50D512 does not support normalize=False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=512,
        )
        model.name = f"EigenPlacesR50D512"

    return model
