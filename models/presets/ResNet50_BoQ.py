import contextlib
import io

import torch


def ResNet50_BoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="resnet50",
            output_dim=16384,
        )

    original_forward = model.forward

    def new_forward(x):
        desc, attn = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model
