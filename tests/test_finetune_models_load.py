import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import _get_weights_path
from models.helper import get_model
from models.presets import PRESET_NAMES
from models.transforms import get_transform


@pytest.mark.parametrize("backbone_arch", ["vitbaset", "vitsmallt"])
@pytest.mark.parametrize("agg_arch", ["salad", "boq", "mixvpr", "salad"])
@pytest.mark.parametrize("image_size", [[224, 224], [322, 322]])
@pytest.mark.parametrize("desc_divider_factor", [1, 2, 4])
def test_available(backbone_arch, agg_arch, image_size, desc_divider_factor):
    model = get_model(
        backbone_arch=backbone_arch,
        agg_arch=agg_arch,
        image_size=image_size,
        desc_divider_factor=desc_divider_factor,
    )
    transform = get_transform(augmentation_level="None", image_size=image_size)
    img = Image.fromarray(
        np.random.randint(
            0, 255, size=(image_size[0], image_size[1], 3), dtype=np.uint8
        )
    )
    inputs = transform(img)
    checkpoint_path = _get_weights_path(
        backbone_arch=backbone_arch,
        agg_arch=agg_arch,
        image_size=image_size,
        desc_divider_factor=desc_divider_factor,
    )
    assert checkpoint_path is not None
    assert isinstance(checkpoint_path, str)
    assert os.path.exists(checkpoint_path)

    model.eval()
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True)["state_dict"], strict=True
    )
    features = model(inputs[None, ...].cuda())
    assert features is not None
    assert features.shape[0] == 1
    assert features.ndim == 2
    assert features.dtype == torch.float32
