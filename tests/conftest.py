import pytest
import torch

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )

def pytest_runtest_setup(item):
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Test requires CUDA")