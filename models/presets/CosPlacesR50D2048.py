import contextlib
import io
import sys

import torch


def CosPlacesR50D2048(noramlize=True):
    if not noramlize:
        raise Exception("CosPlaces does not support noramlize=False")
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
    return model
