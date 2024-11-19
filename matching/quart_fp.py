import faiss
import time
import numpy as np
import torch


def quart_fp(
    global_desc,
    local_desc,
    num_references,
    ground_truth,
    k_values=[1, 5],
    top_k=5,
    **kwargs,
):
    assert top_k >= max(k_values)

    global_desc = global_desc.cpu().numpy()
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    reference_local_desc = local_desc[:num_references]
    query_local_desc = local_desc[num_references:]

    gloabl_index = faiss.IndexFlatIP(reference_desc.shape[1])
    gloabl_index.add(reference_desc)

    start_time = time.time()
    _, _ = gloabl_index.search(query_desc[:1], 1)
    global_search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    dist, predictions = gloabl_index.search(query_desc, top_k)
    predictions = torch.tensor(predictions)
    final_predictions = []
    for i, pred in enumerate(predictions):
        print(i / len(predictions))
        start_time = time.time()

        # Move tensors to GPU if not already
        q_local_desc = query_local_desc[i][None, :, :].cuda()
        reference_local_desc = reference_local_desc.cuda()

        # Ensure tensors are contiguous
        q_local_desc = q_local_desc.contiguous()
        reference_local_desc = reference_local_desc.contiguous()

        # Perform matrix multiplication
        sim = torch.mm(
            q_local_desc.squeeze(),
            reference_local_desc.view(-1, q_local_desc.shape[-1]).T,
        )

        q_matches = torch.argmax(sim, dim=1)
        r_matches = torch.argmax(sim, dim=0)

        mutual_match_count = torch.zeros(top_k)
        for q_idx, r_idx in enumerate(q_matches):
            # Check if the best match for the row also has this row as its best match
            if r_matches[r_idx] == q_idx:
                mutual_match_count[r_idx // sim.shape[1]] += 1

        local_search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        final_predictions.append(
            pred[torch.argsort(mutual_match_count, descending=True)]
        )

    predictions = final_predictions
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break
    correct_at_k = correct_at_k / len(predictions)
    d = {f"R{k}": v for (k, v) in zip(k_values, correct_at_k)}
    return d, torch.stack(predictions), global_search_time + local_search_time
