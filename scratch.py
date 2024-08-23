
#import faiss.contrib.torch_utils
import torch 
import faiss

import numpy as np
from prettytable import PrettyTable

# Define the get_validation_recalls function (provided in the prompt)
def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)
    
    # add references
    faiss_index.add(r_list)

    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(k_values))
    
    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
    
    correct_at_k = correct_at_k / len(predictions)
    d = {k:v for (k,v) in zip(k_values, correct_at_k)}

    if print_results:
        print('\n') # print a new line
        table = PrettyTable()
        table.field_names = ['K']+[str(k) for k in k_values]
        table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performance on {dataset_name}"))
    
    return d, predictions

# Test the get_validation_recalls function


# Generate synthetic data
num_references = 100
num_queries = 10
embedding_size = 128



r_list = np.random.rand(num_references, embedding_size).astype(np.float32)
q_list = np.random.rand(num_queries, embedding_size).astype(np.float32)
print(type(r_list), r_list.dtype)

# Ground truth (for simplicity, assume the first query should match with the first reference, etc.)
gt = [np.array([i]) for i in range(num_queries)]

# Define k_values to evaluate recall
k_values = [1, 5, 10]

# Call the function
results, predictions = get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, dataset_name='Synthetic Test Dataset')

# Output the results and predictions for verification
print("Recall@K results:", results)
print("Predictions:\n", predictions)