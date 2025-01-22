import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helper import get_model
from models.presets import PRESET_NAMES
from models.transforms import get_transform


@pytest.mark.parametrize(
    "backbone_arch", ["vitbase", "vitsmall", "vitbaset", "vitsmallt"]
)
@pytest.mark.parametrize("agg_arch", ["salad", "boq", "mixvpr", "salad"])
@pytest.mark.parametrize("image_size", [[224, 224], [322, 322]])
@pytest.mark.parametrize("desc_divider_factor", [None, 2, 4])
def test_available(backbone_arch, agg_arch, image_size, desc_divider_factor):
    model = get_model(
        backbone_arch=backbone_arch,
        agg_arch=agg_arch,
        image_size=image_size,
        desc_divider_factor=desc_divider_factor,
        normalize=True,
    )
    transform = get_transform(augmentation_level="None", image_size=image_size)
    img = Image.fromarray(
        np.random.randint(
            0, 255, size=(image_size[0], image_size[1], 3), dtype=np.uint8
        )
    )
    inputs = transform(img)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        model = model.cuda()
    features = model(inputs[None, ...])
    assert features is not None
    assert features.shape[0] == 1
    assert features.ndim == 2
    assert features.dtype == torch.float32
    assert torch.allclose(features[0].norm().detach().cpu(), torch.ones(1))




@pytest.mark.parametrize(
    "backbone_arch", ["vitbase", "vitsmall", "vitbaset", "vitsmallt"]
)
@pytest.mark.parametrize("agg_arch", ["salad", "boq", "mixvpr", "salad"])
@pytest.mark.parametrize("image_size", [[224, 224], [322, 322]])
@pytest.mark.parametrize("desc_divider_factor", [None, 2, 4])
def test_available_nonnormalized(backbone_arch, agg_arch, image_size, desc_divider_factor):
    model = get_model(
        backbone_arch=backbone_arch,
        agg_arch=agg_arch,
        image_size=image_size,
        desc_divider_factor=desc_divider_factor,
        normalize=False,
    )
    transform = get_transform(augmentation_level="None", image_size=image_size)
    img = Image.fromarray(
        np.random.randint(
            0, 255, size=(image_size[0], image_size[1], 3), dtype=np.uint8
        )
    )
    inputs = transform(img)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        model = model.cuda()

    features = model(inputs[None, ...])
    assert features is not None
    assert features.shape[0] == 1
    assert features.ndim == 2
    assert features.dtype == torch.float32
    assert not torch.allclose(features[0].norm().detach().cpu(), torch.ones(1))




