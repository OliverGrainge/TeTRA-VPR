import torch
import io 
import contextlib

def CosPlacesR18D32():
    # Suppress both stdout and stderr
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=32,
        )
    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model
