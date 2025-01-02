# test_layer.py

import pytest
import torch

from bitlinear import BitLinear

##############################
# Pytest Fixtures Definitions #
##############################


@pytest.fixture(scope="session")
def device():
    """
    Fixture to determine the device to run tests on.
    Prefers CUDA if available, otherwise falls back to CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def input_tensor(device):
    """
    Fixture to create a single input tensor.
    This tensor is reused across multiple tests to save time.
    """
    tensor = torch.randn(1, 128, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return tensor


@pytest.fixture(scope="session")
def linear_fixture(device):
    """
    Factory fixture to create Linear layer instances.
    Allows creation of layers with or without bias.
    """

    def _create_layer(bias):
        layer = BitLinear(128, 256, bias=bias).to(device)
        layer.eval()  # Initialize in evaluation mode by default
        return layer

    return _create_layer


#################################
# Pytest Test Functions Below   #
#################################


@pytest.mark.parametrize("bias", [True, False])
def test_train_forward(linear_fixture, input_tensor, bias):
    """
    Test the forward pass of the Linear layer in evaluation mode.
    Verifies that the output tensor has the correct shape.
    """
    layer = linear_fixture(bias)
    layer.train()
    out = layer(input_tensor)
    assert out is not None, "Output should not be None."
    assert out.shape == (1, 256), f"Expected output shape (1, 256), got {out.shape}."


@pytest.mark.parametrize("bias", [True, False])
def test_eval_forward(linear_fixture, input_tensor, bias):
    """
    Test the forward pass of the Linear layer in evaluation mode.
    Verifies that the output tensor has the correct shape.
    """
    layer = linear_fixture(bias)
    layer.eval()
    out = layer(input_tensor)
    assert out is not None, "Output should not be None."
    assert out.shape == (1, 256), f"Expected output shape (1, 256), got {out.shape}."


"""

@pytest.mark.cuda
@pytest.mark.parametrize("bias", [True, False])
def test_deploy_forward(linear_fixture, input_tensor, bias):

    layer = linear_fixture(bias)
    layer.deploy()
    out = layer(input_tensor)
    assert out is not None, "Output should not be None."
    assert out.shape == (1, 256), f"Expected output shape (1, 256), got {out.shape}."
"""


@pytest.mark.parametrize("bias", [True, False])
def test_train_eval_consistency(linear_fixture, input_tensor, bias):
    """
    Test the consistency between training and evaluation modes.
    Ensures that the output tensors are close within a specified tolerance.
    """
    layer = linear_fixture(bias)
    layer.train()
    out_train = layer(input_tensor)
    layer.eval()
    out_eval = layer(input_tensor)

    assert (
        out_train.dtype == out_eval.dtype
    ), "Dtype mismatch between train and eval outputs."
    assert torch.allclose(
        out_train, out_eval, rtol=2e-2, atol=2e-2
    ), "Train and eval outputs are not close within the tolerance."


"""
@pytest.mark.parametrize("bias", [True, False])
def test_train_deploy_consistency(linear_fixture, input_tensor, bias):

    layer = linear_fixture(bias)
    layer.train()
    out_train = layer(input_tensor)
    layer.deploy()
    out_eval = layer(input_tensor)
    
    assert out_train.dtype == out_eval.dtype, "Dtype mismatch between train and eval outputs."
    assert torch.allclose(out_train, out_eval, rtol=2e-2, atol=2e-2), \
        "Train and eval outputs are not close within the tolerance."



@pytest.mark.parametrize("bias", [True, False])
def test_eval_deploy_consistency(linear_fixture, input_tensor, bias):

    layer = linear_fixture(bias)
    layer.eval()
    out_train = layer(input_tensor)
    layer.deploy()
    out_eval = layer(input_tensor)
    
    assert out_train.dtype == out_eval.dtype, "Dtype mismatch between train and eval outputs."
    assert torch.allclose(out_train, out_eval, rtol=2e-2, atol=2e-2), \
        "Train and eval outputs are not close within the tolerance."
    
    
"""
"""
@pytest.mark.parametrize("bias", [True, False])
def test_deploy_state_dictionary(linear_fixture, bias):
    layer = linear_fixture(bias)
    layer.deploy()
    state_dict_keys = layer.state_dict().keys()
    
    assert "weight" not in state_dict_keys, "'weight' should not be in state_dict after deployment."
    if bias:
        assert "bias" in state_dict_keys, "'bias' should be in state_dict when bias=True."
    else:
        assert "bias" not in state_dict_keys, "'bias' should not be in state_dict when bias=False."
    assert "qweight" in state_dict_keys, "'weight' should be in state_dict after deployment."
    assert layer.state_dict()["qweight"].dtype == torch.uint8, "Weight should be uint8 after deployment."
    assert "scale" in state_dict_keys, "'scale' should be in state_dict after deployment."

"""


#####################################
# Pytest Configuration for Skipping  #
#####################################


def pytest_runtest_setup(item):
    """
    Hook to skip tests marked with 'cuda' if CUDA is not available.
    """
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping CUDA-dependent tests.")
