import time
from collections import defaultdict

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from prettytable import PrettyTable
from tabulate import tabulate
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from matching.match_cosine import match_cosine
from matching.match_hamming import match_hamming
from models.transforms import get_transform

from dataloaders.eval import memory, latency, accuracy

def get_descriptor_dim(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    """Calculate the dimension of the descriptor output by the model.
    
    Args:
        model: Neural network model
        inputs: Example input tensor
    Returns:
        Dimension of the descriptor
    """
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs)
    return descriptor.shape[-1]

def get_eval_transform(args):
    """Get the evaluation transform based on configuration arguments.
    
    Args:
        args: Configuration arguments containing either preset or image_size
    Returns:
        Transform function to be applied to images during evaluation
    """
    if args.preset is not None:
        return get_transform(preset=args.preset)
    else:
        return get_transform(augmentation_level="None", image_size=args.image_size)



def evaluate(args, model, example_input):
    """Evaluate model performance across multiple metrics.
    
    Args:
        args: Configuration arguments
        model: Model to evaluate
        example_input: Sample input tensor
    Returns:
        Dictionary containing evaluation results
    """
    results = {}
    results["descriptor_dim"] = get_descriptor_dim(model, example_input)
    
    # Determine precision type once
    is_binary = str(model).endswith("t")
    precision = "binary" if is_binary else "float32"

    # Memory evaluations
    if args.model_memory:
        results["model_memory_mb"] = memory.get_model_memory_mb(model)
    
    if args.runtime_memory:
        results["runtime_memory_mb"] = memory.get_runtime_memory_mb(model, example_input)
    
    if args.descriptor_size:
        get_size_fn = memory.get_binary_descriptor_size_bytes if is_binary else memory.get_floating_descriptor_size_bytes
        results["descriptor_size_bytes"] = get_size_fn(model, example_input)

    # Latency evaluations  
    if args.feature_extraction_latency:
        results["feature_extraction_latency_ms"] = latency.get_model_inference_latency_ms(model, example_input)
    
    if args.retrieval_latency:
        get_latency_fn = latency.get_binary_retrieval_latency if is_binary else latency.get_floating_retrieval_latency
        results["retrieval_latency_ms"] = get_latency_fn(results["descriptor_dim"])

    # Accuracy evaluations
    k_values = [1]
    if args.accuracy and len(args.val_set_names) > 0:
        transform = get_eval_transform(args)
        for val_set_name in args.val_set_names:
            dataset = accuracy.get_val_dataset(val_set_name, args.val_dataset_dir, transform)
            desc = accuracy.compute_descriptors(model, dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            
            recalls = accuracy.get_recall_at_k(desc, dataset, precision=precision, k_values=k_values)
            for idx, k in enumerate(recalls):
                results[f"{repr(dataset)}_R@{k_values[idx]}"] = recalls[idx]

            if args.dataset_retrieval_latency == 1:
                get_latency_fn = latency.get_binary_retrieval_latency if is_binary else latency.get_floating_retrieval_latency
                results[f"{repr(dataset)}_retrieval_latency"] = get_latency_fn(
                    results["descriptor_dim"], 
                    ref_n=len(dataset)
                )
    return results





    



