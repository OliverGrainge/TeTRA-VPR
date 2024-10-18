import torch


def ResNet50_BoQ():
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="resnet50", output_dim=16384)

    original_forward = model.forward
    def new_forward(x):
        desc, attn = original_forward(x)
        return {"global_desc": desc, "local_desc": None, "attention": attn}
    model.forward = new_forward
    return model