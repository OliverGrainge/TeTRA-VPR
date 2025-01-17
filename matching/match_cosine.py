import time

import faiss
import numpy as np


def match_cosine(
    desc: "torch.Tensor",
    num_references: int,
    ground_truth: "np.ndarray",
    k_values: list[int] = [1, 5, 10]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarity matching accuracy using FAISS.

    Args:
        desc: Input descriptors (reference + query)
        num_references: Number of reference descriptors
        ground_truth: Ground truth matches for queries
        k_values: Values of k for accuracy computation (default: [1, 5, 10])

    Returns:
        Tuple of (accuracy percentages at each k, predicted matches)
    """
    # Convert to numpy and ensure contiguous memory layout for better performance
    desc = desc.cpu().numpy().astype(np.float32, copy=False)
    
    # Input validation
    if not isinstance(k_values, (list, np.ndarray)) or not k_values:
        raise ValueError("k_values must be a non-empty list or array")
    if num_references >= len(desc):
        raise ValueError("num_references must be less than total descriptors")
    
    # Split descriptors into reference and query sets
    reference_desc = desc[:num_references]
    query_desc = desc[num_references:]
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(reference_desc)
    faiss.normalize_L2(query_desc)
    
    # Create and configure FAISS index
    index = faiss.IndexFlatIP(reference_desc.shape[1])
    index.add(reference_desc)
    
    # Perform similarity search
    max_k = max(k_values)
    dist, predictions = index.search(query_desc, max_k)
    
    # Calculate accuracy for each k value
    correct_at_k = np.zeros(len(k_values), dtype=np.float32)
    for q_idx, pred in enumerate(predictions):
        # Use vectorized operations for better performance
        found_at = np.where(np.in1d(pred, ground_truth[q_idx]))[0]
        if len(found_at) > 0:
            first_match = found_at[0]
            correct_at_k += (first_match < np.array(k_values))
    
    # Calculate percentage
    correct_at_k = (correct_at_k / len(predictions)) * 100
    return correct_at_k, predictions
