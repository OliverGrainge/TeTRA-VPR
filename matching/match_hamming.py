import time

import faiss
import numpy as np


def match_hamming(
    desc: np.ndarray,
    num_references: int,
    ground_truth: np.ndarray,
    k_values: list[int] = [1, 5, 10],
) -> tuple[np.ndarray, np.ndarray]:
    """Match descriptors using Hamming distance with Faiss.

    Args:
        desc: Input descriptors as a numpy array
        num_references: Number of reference descriptors
        ground_truth: Ground truth matches for evaluation
        k_values: List of k values for k-nearest neighbor evaluation

    Returns:
        tuple containing:
            - Percentage of correct matches at different k values
            - Predicted indices for each query

    Raises:
        ValueError: If input parameters are invalid
    """

    def float_to_binary_desc(desc: np.ndarray) -> np.ndarray:
        """Convert float descriptors to binary packed format."""
        binary = (desc > 0).astype(np.bool_)
        n_bytes = (binary.shape[1] + 7) // 8
        return np.packbits(binary, axis=1)[:, :n_bytes]

    # Input validation
    if not k_values:
        raise ValueError("k_values list cannot be empty")
    if num_references >= len(desc):
        raise ValueError("num_references must be less than total descriptors")
    if not all(k > 0 for k in k_values):
        raise ValueError("All k values must be positive")

    # Convert to numpy if not already
    if not isinstance(desc, np.ndarray):
        desc = desc.cpu().numpy()

    # Convert to binary format
    desc = float_to_binary_desc(desc)

    # Split into reference and query
    reference_desc = desc[:num_references]
    query_desc = desc[num_references:]

    # Create and configure Faiss index
    bits_per_vector = reference_desc.shape[1] * 8
    index = faiss.IndexBinaryFlat(bits_per_vector)

    # Add reference descriptors to index
    index.add(reference_desc)

    # Perform search
    max_k = max(k_values)
    distances, predictions = index.search(query_desc, max_k)

    # Calculate accuracy at different k values
    correct_at_k = np.zeros(len(k_values), dtype=np.float32)
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = (correct_at_k / len(predictions)) * 100
    return correct_at_k, predictions
