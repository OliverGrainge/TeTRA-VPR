import torch


def ResNet50_BoQ():
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="resnet50", output_dim=16384)
    return model