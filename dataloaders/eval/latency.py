import time
from typing import List, Tuple

import faiss
import numpy as np
import torch


def get_model_inference_latency_ms(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    num_warmup: int = 100,
    num_samples: int = 500,
) -> Tuple[float, float]:
    """
    Measure model inference latency in milliseconds.

    Args:
        model: PyTorch model to evaluate
        inputs: Input tensor for the model
        num_warmup: Number of warmup runs
        num_samples: Number of timing samples to collect

    Returns:
        mean_latency_ms: Mean latency in milliseconds
    """

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs)

    samples = []
    for _ in range(num_samples):
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        samples.append((end - start) * 1000.0)

    return np.mean(samples)


def float_to_binary_desc(desc: np.ndarray) -> np.ndarray:
    """Convert floating point descriptors to binary packed format."""
    binary = (desc > 0).astype(np.bool_)
    n_bytes = (binary.shape[1] + 7) // 8
    return np.packbits(binary, axis=1)[:, :n_bytes]


def _measure_search_latency(
    index: faiss.Index, query: np.ndarray, num_warmup: int = 3, num_samples: int = 5
) -> float:
    """Helper function to measure search latency for both binary and floating point indices."""
    # Warmup
    for _ in range(num_warmup):
        _ = index.search(query, k=1)

    samples = []
    for _ in range(num_samples):
        start = time.perf_counter()
        _ = index.search(query, k=1)
        samples.append((time.perf_counter() - start) * 1000.0)

    return np.mean(samples)


def get_binary_retrieval_latency(desc_size: int, ref_n: int = 10000) -> float:
    """Measure Hamming distance-based retrieval latency in milliseconds."""
    query = np.random.randn(1, desc_size)
    reference = np.random.randn(ref_n, desc_size)
    binary_query = float_to_binary_desc(query)
    binary_reference = float_to_binary_desc(reference)

    index = faiss.IndexBinaryFlat(binary_reference.shape[1] * 8)
    index.add(binary_reference)

    return _measure_search_latency(index, binary_query)


def get_floating_retrieval_latency(desc_size: int, ref_n: int = 10000) -> float:
    """Measure floating point inner product retrieval latency in milliseconds."""
    query = np.random.randn(1, desc_size)
    reference = np.random.randn(ref_n, desc_size)

    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(reference)

    return _measure_search_latency(index, query)
