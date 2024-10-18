import torch 
import numpy as np 
import faiss 




def global_cosine_sim(global_desc, num_references, ground_truth, k_values=[1, 5]):
    global_desc = global_desc.cpu().numpy()
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    index = faiss.IndexFlatIP(reference_desc.shape[1])
    index.add(reference_desc)
    print("=================== Searching ===================")
    dist, predictions = index.search(query_desc, max(k_values))
    print("=================== Finished Searching ===================")

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        print(q_idx/predictions.shape[0])
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}
    print(d)
    return d, predictions