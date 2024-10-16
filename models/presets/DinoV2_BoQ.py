import torch


def DinoV2_BoQ():
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="dinov2", output_dim=12288)
    return model


