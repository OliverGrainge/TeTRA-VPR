from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from tabulate import tabulate
from prettytable import PrettyTable
from matching.match_hamming import match_hamming
from matching.match_cosine import match_cosine
import time
from torch.cuda.amp import autocast
import faiss
from matching.match_cosine import match_cosine


def get_sample_output(model, transform):
    model.eval()
    sample = np.random.randint(0, 255, size=(224, 224, 3))
    sample = Image.fromarray(sample.astype(np.uint8))
    sample = transform(sample)
    sample = sample.unsqueeze(0)
    output = model(sample)
    return output


def get_model_memory(model, inputs):
    # Sum up parameter memory
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel() * param.element_size()
        
    # Convert bytes to MB
    return total_params / (1024 * 1024)


def get_runtime_memory(model, inputs):
    # Assumes GPU usage. If on CPU, consider psutil or skip.
    
    if torch.cuda.is_available():
        samples = []
        for _ in range(5):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Forward pass with mixed precision
            model.eval()
            with torch.no_grad():
                with autocast():
                    _ = model(inputs)

            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        samples += [peak_memory]
        return np.mean(peak_memory)
        
    else:
        # On CPU, we can't easily measure memory here without external tools.
        return 0.0


def get_model_inference_latency(model, inputs):
    model.eval()
    # warmup
    for _ in range(3): 
        _ = model(inputs)
    samples = []
    for _ in range(20):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure timing is accurate on GPU
        with torch.inference_mode():
            _ = model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        latency_ms = (end - start) * 1000.0
        samples.append(latency_ms)
    
    return np.mean(samples)

def float_to_binary_desc(desc):
    # Convert floating-point descriptors to binary and pack into bytes
    binary = (desc > 0).astype(np.bool_)
    # Calculate number of bytes needed (round up to multiple of 8)
    n_bytes = (binary.shape[1] + 7) // 8
    packed = np.packbits(binary, axis=1)[:, :n_bytes]
    return packed

def get_hamming_retrieval_latency(desc_size, ref_n=10000):
    query = np.random.randn(1, desc_size)
    reference = np.random.randn(ref_n, desc_size)
    binary_query = float_to_binary_desc(query)
    binary_reference = float_to_binary_desc(reference)

    index = faiss.IndexBinaryFlat(
        binary_reference.shape[1] * 8
    )
    index.add(binary_reference)

    #warmup 
    for _ in range(3):
        _, _ = index.search(binary_query, 1)
    
    samples = []
    for _ in range(5): 
        start_time = time.time()
        _, _, index.search(binary_query, 1)
        samples.append((time.time() - start_time) * 1000.0)
    return np.mean(samples)



def get_floating_retrieval_latency(desc_size, ref_n=10000): 
    query = np.random.randn(1, desc_size)
    reference = np.random.randn(ref_n, desc_size)

    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(reference)
    #warmup 
    for _ in range(3):
        _, _ = index.search(query, 1)
    
    samples = []
    for _ in range(5): 
        start_time = time.time()
        _, _, index.search(query, 1)
        samples.append((time.time() - start_time) * 1000.0)
    return np.mean(samples)



def get_floating_descriptor_size(model, inputs):
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs)
        descriptor = descriptor["global_desc"]
    # descriptor is a torch.Tensor
    size_in_bytes = descriptor.numel() * descriptor.element_size()
    return size_in_bytes


def get_binary_descriptor_size(model, inputs):
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs)
        descriptor = descriptor["global_desc"]
    # descriptor is a torch.Tensor
    size_in_bytes = (descriptor.numel() * descriptor.element_size())/8
    return size_in_bytes



def get_descriptor_dim(model, inputs):
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs)
        descriptor = descriptor["global_desc"]
    # Assuming descriptor is 1D or has a single trailing dimension that represents the embedding dimension
    # If the output shape is something like (batch_size, D), we take D
    dim = descriptor.shape[-1]
    return dim

class VPREval(pl.LightningModule):
    def __init__(
        self,
        model,
        transform,
        val_set_names=["pitts30k"],
        val_dataset_dir=None, 
        batch_size=32,
        num_workers=4,
        matching_function=match_cosine,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.matching_function = matching_function
        self.val_dataset_dir = val_dataset_dir

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset30k
                    self.val_datasets.append(
                        PittsburghDataset30k(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "pitts250k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset250k
                    self.val_datasets.append(
                        PittsburghDataset250k(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "msls" in val_set_name.lower():
                    from dataloaders.val.MapillaryDataset import MSLS
                    self.val_datasets.append(
                        MSLS(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="val")
                    )
                elif "nordland" in val_set_name.lower():
                    from dataloaders.val.NordlandDataset import NordlandDataset
                    self.val_datasets.append(
                        NordlandDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "sped" in val_set_name.lower():
                    from dataloaders.val.SPEDDataset import SPEDDataset
                    self.val_datasets.append(
                        SPEDDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test"))
                elif "essex" in val_set_name.lower():
                    from dataloaders.val.EssexDataset import EssexDataset
                    self.val_datasets.append(
                        EssexDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "sanfrancicscosmall" in val_set_name.lower():
                    from dataloaders.val.SanFranciscoSmall import SanFranciscoSmall
                    self.val_datasets.append(
                        SanFranciscoSmall(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "tokyo" in val_set_name.lower():
                    from dataloaders.val.Tokyo247 import Tokyo247
                    self.val_datasets.append(
                        Tokyo247(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "cross" in val_set_name.lower():
                    from dataloaders.val.CrossSeasonDataset import CrossSeasonDataset
                    self.val_datasets.append(
                        CrossSeasonDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    shuffle=False,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
            )
        return val_dataloaders

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        return x

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = {}
        for name in self.val_set_names:
            self.validation_outputs[name] = defaultdict(list)

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(
                value.detach().cpu()
            )
        return descriptors["global_desc"].detach().cpu()
    

    
    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""

        # Initialize tables
        accuracy_table = PrettyTable()
        accuracy_table.field_names = ["Dataset", "Search Function", "R@1"]
        accuracy_table.align["Dataset"] = "l"
        accuracy_table.align["Search Function"] = "l"
        accuracy_table.align["R@1"] = "r"

        resource_table = PrettyTable()
        resource_table.field_names = ["Metric", "Value", "Unit"]
        resource_table.align["Metric"] = "l"
        resource_table.align["Value"] = "r"
        resource_table.align["Unit"] = "l"

        # Iterate through each validation set
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]

            # Concatenate validation outputs
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            # Perform cosine similarity matching
            fp_recalls_dict, _, cosine_search_time = match_cosine(
                **set_outputs, 
                num_references=val_dataset.num_references,
                ground_truth=val_dataset.ground_truth,
                k_values=[1, 5, 10]
            )

            # Extract dataset name (cleaning up the __repr__ if necessary)
            dataset_repr = val_dataset.__repr__()
            dataset_name = " ".join(dataset_repr.split("_")[:-1]) if "_" in dataset_repr else dataset_repr

            # Update accuracy table for cosine similarity
            accuracy_table.add_row([dataset_name, "Cosine", f"{fp_recalls_dict['R1']:.2f}"])

            # Perform hamming distance matching
            q_recalls_dict, _, hamming_search_time = match_hamming(
                **set_outputs, 
                num_references=val_dataset.num_references,
                ground_truth=val_dataset.ground_truth,
                k_values=[1, 5, 10]
            )

            # Update accuracy table for hamming distance
            accuracy_table.add_row([dataset_name, "Hamming", f"{q_recalls_dict['R1']:.3f}"])

        

        # Generate a dummy input image and transform it
        img = torch.randint(0, 255, size=(224, 224, 3), dtype=torch.uint8).numpy()
        img = Image.fromarray(img)
        inputs = self.transform(img).unsqueeze(0).to(next(self.model.parameters()).device)

        # Collect resource metrics
        feature_latency = get_model_inference_latency(self.model, inputs)
        
        model_memory = get_model_memory(self.model, inputs)
        runtime_memory = get_runtime_memory(self.model, inputs)
        descriptor_size_floating = get_floating_descriptor_size(self.model, inputs)
        descriptor_size_binary = get_binary_descriptor_size(self.model, inputs)
        descriptor_dim = get_descriptor_dim(self.model, inputs)
        search_latency_hamming = get_hamming_retrieval_latency(descriptor_dim)
        search_latency_fp = get_floating_retrieval_latency(descriptor_dim)

        resource_table.add_row(["Feature Extraction Latency", f"{feature_latency:.2f}", "ms"])
        resource_table.add_row(["Search Latency Floating", f"{search_latency_fp:.2f}", "ms/10k"])
        resource_table.add_row(["Search Latency Hamming", f"{search_latency_hamming:.2f}", "ms/10k"])
        resource_table.add_row(["Model Memory", f"{model_memory:.2f}", "Mb"])
        resource_table.add_row(["Peak Runtime Memory", f"{runtime_memory:.2f}", "Mb"])
        resource_table.add_row(["Descriptor Dimension", f"{descriptor_dim}", "#"])
        resource_table.add_row(["Descriptor Size Floating", f"{descriptor_size_floating}", "bytes"])
        resource_table.add_row(["Descriptor Size binary", f"{descriptor_size_binary}", "bytes"])
        


        # Print accuracy results
        print(" ")
        print(" ")
        print(" ")
        print("\n" + "="*50)
        print("                 ACCURACY RESULTS")
        print("="*50)
        print(accuracy_table)

        # Print resource results
        print("\n" + "="*50)
        print("                 RESOURCE RESULTS")
        print("="*50)
        print(resource_table)
        print(" ")
        print(" ")
        print(" ")

        return None

