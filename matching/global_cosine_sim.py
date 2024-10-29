import faiss
import numpy as np
import torch
import time

def global_cosine_sim(global_desc, num_references, ground_truth, k_values=[1, 5, 10]):
    global_desc = global_desc.cpu().numpy()
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]
    
    index = faiss.IndexFlatIP(reference_desc.shape[1])
    index.add(reference_desc)

    dist, predictions = index.search(query_desc, max(k_values))

    start_time = time.time()
    _, _ = index.search(query_desc, 1)
    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {f"R{k}": v for (k, v) in zip(k_values, correct_at_k)}
    return d, predictions, search_time
