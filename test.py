import torch
import time
from models.helper import get_model 
from tqdm import tqdm

def benchmark_inference_latency(model: torch.nn.Module, input_tensor: torch.Tensor, num_iterations: int = 1000) -> float:
    """
    Benchmark mean inference latency of a PyTorch model on CUDA.
    Returns mean latency in milliseconds.
    """
    model = model.to("cuda")
    input_tensor = input_tensor.to("cuda")
    
    # Warm-up
    with torch.no_grad():
        for _ in tqdm(range(200)):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations)):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000)  # ms
    
    return sum(latencies) / len(latencies)

model = get_model(backbone_arch="vitsmallt", agg_arch="boq", image_size=[224, 224], desc_divider_factor=1)
model = model.cuda()
input_tensor = torch.randn(1, 3, 224, 224).cuda()

baseline = benchmark_inference_latency(model, input_tensor)

model.deploy(use_bitblas=False)
baseline_deployed = benchmark_inference_latency(model, input_tensor)

print(f"Baseline latency: {baseline:.2f} ms")
print(f"Deployed latency: {baseline_deployed:.2f} ms")




