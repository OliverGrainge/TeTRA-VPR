import contextlib
import io

import torch


def EigenPlacesR18D512():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=512,
        )

    return model
