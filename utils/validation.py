import numpy as np
import faiss
import faiss.contrib.torch_utils
import torch
from prettytable import PrettyTable
from usearch.index import Index

        

def float32_search(r_list, q_list, k_values, faiss_gpu=False):
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
    r_list = r_list.float()
    q_list = q_list.float()
    faiss_index.add(r_list)

    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(k_values))
    return predictions




def binary_search(r_list, q_list, k_values, faiss_gpu=False):
    def binary_quantize(tensor): 
        qtensor = torch.zeros_like(tensor, dtype=torch.uint8)
        pos_mask = tensor >= 0
        qtensor[pos_mask] = 1
        return np.packbits(qtensor.detach().numpy(), axis=1)
    

    qr_list, qq_list = binary_quantize(r_list), binary_quantize(q_list)

    index = faiss.IndexBinaryFlat(qr_list.shape[1] * 8)
    index.add(qr_list)
    _, predictions = index.search(qq_list, max(k_values))
    return predictions

     


def int8_search(r_list, q_list, k_values, faiss_gpu=False):
    def int8_quantize(tensor, scale=None):
        tensor = tensor.detach().float()
        if scale is None:
            max_val = np.max(np.abs(tensor))  # Use the maximum absolute value for symmetric quantization
            scale = max_val / 127.0  # 127 is the maximum value of int8

        # Quantize the tensor (zero point is 0 in symmetric quantization)
        quantized_tensor = np.round(tensor / scale)

        # Clip the values to be within the int8 range [-127, 127]
        quantized_tensor = np.clip(quantized_tensor, -127, 127).astype(np.int8)

        return quantized_tensor, scale
     
    qr_list, scale = int8_quantize(r_list)
    qq_list = int8_quantize(q_list, scale)
    dim = qr_list.shape[1]
    index = Index(ndim=dim, metric='cos', dtype='i8')

    for i, embedding in enumerate(qr_list):
        index.add(i, embedding)

    matches = index.search(qr_list, max(k_values))

    arr = []
    for match in matches: 
        arr.append(np.array(match.to_list())[:, 0])
    return np.array(arr)


          


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?', precision='float32'):
        if precision == 'float32':
             predictions = float32_search(r_list, q_list, k_values, faiss_gpu=faiss_gpu)
        elif precision == 'binary':
             predictions = binary_search(r_list, q_list, k_values, faiss_gpu=faiss_gpu)
        elif precision == "int8":
             predictions = int8_search(r_list, q_list, k_values, faiss_gpu=faiss_gpu)
             print(predictions)
        else: 
             raise Exception(f"type {precision} is not supported")
        
        
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