import time

import faiss
import numpy as np


def float_to_binary_desc(desc):
    # Convert floating-point descriptors to binary and pack into bytes
    binary = (desc > 0).astype(np.bool_)
    # Calculate number of bytes needed (round up to multiple of 8)
    n_bytes = (binary.shape[1] + 7) // 8
    packed = np.packbits(binary, axis=1)[:, :n_bytes]
    return packed


def match_hamming(global_desc, num_references, ground_truth, k_values=[1, 5, 10]):
    global_desc = global_desc.cpu().numpy()
    # Convert to packed binary descriptors
    global_desc = float_to_binary_desc(global_desc)
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    # Create Faiss index for Hamming distance search
    index = faiss.IndexBinaryFlat(
        reference_desc.shape[1] * 8
    )  # *8 since each byte contains 8 bits
    index.add(reference_desc)

    distances, predictions = index.search(query_desc, max(k_values))
    # Search using Faiss
    start_time = time.time()
    _, _ = index.search(query_desc, 1)
    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = (correct_at_k / len(predictions)) * 100
    d = {f"R{k}": v for (k, v) in zip(k_values, correct_at_k)}
    return d, predictions, search_time
