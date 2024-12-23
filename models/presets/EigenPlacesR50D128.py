import torch
import io 
import contextlib



def EigenPlacesR50D128():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=128,
        )

    original_forward = model.forward

    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc}

    model.forward = new_forward
    return model


