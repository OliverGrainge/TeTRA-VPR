import pytest
import torch 
import torch.nn as nn 
import sys 
import os 
from PIL import Image
import numpy as np 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.presets import PRESET_NAMES 
from models.helper import get_model
from models.transforms import get_transform

@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_available(preset_name): 
    model = get_model(preset=preset_name)
    transform = get_transform(preset=preset_name)
    assert model is not None
    assert transform is not None




@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_feature_extraction(preset_name): 
    model = get_model(preset=preset_name)
    transform = get_transform(preset=preset_name)
    image = Image.fromarray(np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8))
    inputs = transform(image)
    features = model(inputs[None, ...])
    assert features is not None
    assert features.shape[0] == 1 
    assert features.ndim == 2 
    assert features.dtype == torch.float32


