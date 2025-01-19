import contextlib
import io

import torch


class ResNet50_BoQ_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="resnet50",
            output_dim=16384,
        )

    def forward(self, x):
        return self.model(x)[0]


def ResNet50_BoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = ResNet50_BoQ_model()
    return model
