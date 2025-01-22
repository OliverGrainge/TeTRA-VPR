import contextlib
import io

import torch


def EigenPlacesR18D256(normalize=True):
    if not normalize:
        raise Exception("EigenPlacesR18D256 does not support normalize=False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=256,
        )

    return model
