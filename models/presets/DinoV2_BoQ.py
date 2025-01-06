import contextlib
import io

import torch


def DinoV2_BoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
        )

    # Modify the forward method to return only the first element of the tuple
    original_forward = model.forward

    def new_forward(x):
        desc, attn = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward

    return model
