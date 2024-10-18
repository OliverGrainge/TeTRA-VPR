import torch


def CosPlaces():
    model = torch.hub.load(
        "gmberton/cosplace",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )
    original_forward = model.forward
    def new_forward(x):
        desc = original_forward(x)
        return {"global_desc": desc, "local_desc": None}
    model.forward = new_forward
    return model
