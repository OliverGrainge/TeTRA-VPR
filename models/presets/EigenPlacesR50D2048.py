import contextlib
import io

import torch


def EigenPlacesR50D2048(normalize=True):
    if not normalize:
        raise Exception("EigenPlacesR50D2048 does not support normalize=False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )

    return model
