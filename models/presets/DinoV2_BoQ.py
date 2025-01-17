import contextlib
import io

import torch


class DinoV2_BoQ_model(torch.nn.Module):
    def __init__(self):
        super(DinoV2_BoQ_model, self).__init__()
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
        )

    def forward(self, x):
        return self.model(x)[0]

def DinoV2_BoQ():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        model = DinoV2_BoQ_model()
    return model
