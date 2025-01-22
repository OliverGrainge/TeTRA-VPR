import contextlib
import io

import torch


def CosPlacesR18D64(normalize=True):
    if not normalize:
        raise Exception("CosPlacesR18D64 does not support normalize=False")
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
    model.name = f"CosPlacesR18D64"
    return model
