import contextlib
import io

import torch


def CosPlacesR50D128(normalize=True):
    if not normalize:
        raise Exception("CosPlacesR50D128 does not support normalize=False")
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
    return model
