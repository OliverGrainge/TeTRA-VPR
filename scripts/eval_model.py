import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import yaml



from models.helper import get_model
from NeuroCompress.NeuroPress import freeze_model

with open("../config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

backbone_arch = "ternary_vit_base"
agg_arch = "cls"


def measure_memory(model, verbose=False):
    """
    Calculates the memory consumption of a PyTorch model based on its state_dict.

    Args:
        model (torch.nn.Module): The PyTorch model.
        verbose (bool): If True, prints memory usage for each parameter and buffer.

    Returns:
        total_memory (str): Total memory consumption in a human-readable format.
    """
    state_dict = model.state_dict()
    total_bytes = 0
    for name, tensor in state_dict.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        total_bytes += tensor_bytes
        # if verbose:
        # print(f"{name}: {tensor_bytes / (1024 ** 2):.2f} MB")
    if verbose:
        print(f"Total Memory: {total_bytes / (1024 ** 2):.2f} MB")

    return total_bytes / (1024**2)


def measure_latency(
    model, input_data, warmup_runs=10, timed_runs=100, device=torch.device("cuda")
):
    """
    Measures the average latency of a PyTorch model on CUDA.

    Args:
        model (torch.nn.Module): The PyTorch model to be measured.
        input_data (torch.Tensor or tuple/list of torch.Tensor):
            Input tensor(s) for the model. If the model accepts multiple inputs,
            provide them as a tuple or list.
        warmup_runs (int): Number of warm-up runs to perform before timing.
        timed_runs (int): Number of runs to time for latency measurement.
        device (torch.device): The device on which to perform the measurement.

    Returns:
        float: The average latency in milliseconds.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available. Please ensure a CUDA-enabled device is present."
        )

    model = model.to(device)
    model.eval()

    # Move input data to the specified device
    if isinstance(input_data, (list, tuple)):
        input_data = [inp.to(device) for inp in input_data]
    else:
        input_data = input_data.to(device)

    # Ensure no gradients are being tracked
    with torch.no_grad():
        # Warm-up phase
        for _ in range(warmup_runs):
            if isinstance(input_data, (list, tuple)):
                _ = model(*input_data)
            else:
                _ = model(input_data)
        torch.cuda.synchronize()  # Wait for all warm-up operations to finish

        # Timing phase
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(timed_runs):
            if isinstance(input_data, (list, tuple)):
                _ = model(*input_data)
            else:
                _ = model(input_data)

        end_event.record()

        # Wait for all timed operations to finish
        torch.cuda.synchronize()

        # Calculate elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_latency = elapsed_time_ms / timed_runs
    print(f"Avg Latency: {avg_latency} ms")

    return avg_latency


model = get_model(
    image_size=[224, 224],
    backbone_arch=backbone_arch,
    agg_arch=agg_arch,
    model_config=config["Model"],
    normalize_output=True,
)
img = torch.randn(1, 3, 224, 224)

measure_memory(model, verbose=True)
freeze_model(model)
measure_memory(model, verbose=True)


tern_lat = []
fp_lat = []
x = [1, 2, 4, 8, 16, 32]
for batch_size in x:
    img = torch.randn(batch_size, 3, 224, 224)
    model = get_model(
        image_size=[224, 224],
        backbone_arch=backbone_arch,
        agg_arch=agg_arch,
        model_config=config["Model"],
        normalize_output=True,
    )
    fp_lat.append(measure_latency(model, img))
    freeze_model(model)
    tern_lat.append(measure_latency(model, img))
    print("==")


import matplotlib.pyplot as plt

plt.plot(x, fp_lat, label="full precision")
plt.plot(x, tern_lat, label="ternary")
plt.legend()
plt.show()
