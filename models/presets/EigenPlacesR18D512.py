import torch
import io 
import contextlib

def EigenPlacesR18D512():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=512,
        )

    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model

