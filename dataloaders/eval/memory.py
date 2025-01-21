import numpy as np
import torch


def get_model_memory_mb(model) -> float:
    """Calculate static memory usage of model parameters and buffers in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Memory usage in MB
    """

    def _get_tensor_memory(tensor):
        return tensor.numel() * tensor.element_size()

    param_memory = sum(_get_tensor_memory(p) for p in model.parameters())
    buffer_memory = sum(_get_tensor_memory(b) for b in model.buffers())

    return (param_memory + buffer_memory) / (1024 * 1024)


def get_runtime_memory_mb(model, inputs, num_trials: int = 5) -> float:
    """Measure peak runtime memory usage during model inference in MB.

    Args:
        model: PyTorch model
        inputs: Model inputs
        num_trials: Number of trials to average over

    Returns:
        float: Average peak memory usage in MB
    """
    if not torch.cuda.is_available():
        return 0.0

    def _run_inference():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(inputs)
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Warm-up run
    with torch.no_grad():
        _ = model(inputs)
    torch.cuda.empty_cache()

    # Measure memory over multiple trials
    samples = [_run_inference() for _ in range(num_trials)]
    return np.mean(samples)


def get_floating_descriptor_size_bytes(
    model: torch.nn.Module, inputs: torch.Tensor
) -> int:
    """Calculate the size of the model's floating-point descriptor output in bytes.

    Args:
        model: PyTorch model that outputs a descriptor tensor
        inputs: Input tensor to the model

    Returns:
        int: Size of the descriptor in bytes

    Raises:
        TypeError: If model output is not a tensor
    """
    with torch.no_grad():
        descriptor = model(inputs)

    if not isinstance(descriptor, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(descriptor)}")

    return descriptor[0].numel() * descriptor[0].element_size()


def get_binary_descriptor_size_bytes(
    model: torch.nn.Module, inputs: torch.Tensor
) -> float:
    """Calculate the size of the model's binary descriptor output in bytes.
    Assumes the descriptor can be binarized (1 bit per element instead of 32/64 bits).

    Args:
        model: PyTorch model that outputs a descriptor tensor
        inputs: Input tensor to the model

    Returns:
        float: Size of the binary descriptor in bytes

    Raises:
        TypeError: If model output is not a tensor
    """
    with torch.no_grad():
        descriptor = model(inputs)

    if not isinstance(descriptor, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(descriptor)}")

    bits_per_element = 1  # Binary descriptor uses 1 bit per element
    total_bits = descriptor.numel() * bits_per_element
    return total_bits / 8  # Convert bits to bytes
