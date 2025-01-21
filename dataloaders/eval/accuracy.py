from typing import List, Optional, Union

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from matching.match_cosine import match_cosine
from matching.match_hamming import match_hamming

# Define a mapping of dataset names to their corresponding classes
DATASET_MAPPING = {
    "pitts30k": ("dataloaders.val.PittsburghDataset", "PittsburghDataset30k"),
    "pitts250k": ("dataloaders.val.PittsburghDataset", "PittsburghDataset250k"),
    "msls": ("dataloaders.val.MapillaryDataset", "MSLS"),
    "nordland": ("dataloaders.val.NordlandDataset", "NordlandDataset"),
    "sped": ("dataloaders.val.SPEDDataset", "SPEDDataset"),
    "essex": ("dataloaders.val.EssexDataset", "EssexDataset"),
    "sanf": ("dataloaders.val.SanFranciscoDataset", "SanFrancisco"),
    "tokyo": ("dataloaders.val.Tokyo247Dataset", "Tokyo247"),
    "cross": ("dataloaders.val.CrossSeasonDataset", "CrossSeasonDataset"),
    "amstertime": ("dataloaders.val.AmsterTimeDataset", "AmsterTime"),
    "eynsham": ("dataloaders.val.EynshamDataset", "Eynsham"),
    "svox": ("dataloaders.val.SVOXDataset", "SVOX"),
}


def get_val_dataset(val_set_name: str, val_dataset_dir: str, transform):
    val_set_name = val_set_name.lower()

    # Find the first matching dataset key
    dataset_key = next(
        (key for key in DATASET_MAPPING.keys() if key in val_set_name), None
    )

    if dataset_key is None:
        raise NotImplementedError(f"Evaluation set {val_set_name} not implemented")

    # Import and instantiate the dataset class
    module_path, class_name = DATASET_MAPPING[dataset_key]
    dataset_module = __import__(module_path, fromlist=[class_name])
    dataset_class = getattr(dataset_module, class_name)

    return dataset_class(
        val_dataset_dir=val_dataset_dir,
        input_transform=transform,
        which_set="test" if "msls" not in val_set_name else "val",
    )


def compute_descriptors(
    model: nn.Module, dataset: Dataset, batch_size: int = 32, num_workers: int = 4
) -> Tensor:
    """Compute descriptors for all images in the dataset using the given model.

    Args:
        model (nn.Module): Neural network model for computing descriptors
        dataset (Dataset): Dataset containing images
        batch_size (int, optional): Batch size for processing. Defaults to 32
        num_workers (int, optional): Number of worker processes. Defaults to 4

    Returns:
        Tensor: Concatenated descriptors for all images
    """
    device = next(model.parameters()).device  # Get model's device

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    all_desc = []
    device = next(model.parameters()).device
    for batch in tqdm.tqdm(dataloader, desc=f"Computing Descriptors: {str(dataset).replace('_test', '')}"):
        imgs, _ = batch
        imgs = imgs.to(device)  # Move images to correct device
        if device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                desc = model(imgs.to(device))
        else:
            with torch.inference_mode():
                desc = model(imgs.to(device))
        all_desc.append(desc.cpu())  # Move to CPU before converting to numpy

    return torch.cat(all_desc, dim=0)


def get_recall_at_k(
    desc: Tensor,
    dataset: Dataset,
    k_values: Union[List[int], tuple[int, ...]] = [1, 5, 10],
    precision: str = "float32",
) -> List[float]:
    """Calculate recall@k metrics for the given descriptors.

    Args:
        desc (Tensor): Image descriptors
        dataset (Dataset): Dataset containing validation data and ground truth
        k_values (Union[List[int], tuple[int, ...]], optional): Values of k for recall calculation.
            Defaults to [1, 5, 10]
        precision (str, optional): Precision type ('float32' or 'binary'). Defaults to 'float32'

    Returns:
        List[float]: Recall values for each k

    Raises:
        ValueError: If precision type is not supported
    """
    if not isinstance(k_values, list):
        k_values = list(k_values)

    matching_fn = {"float32": match_cosine, "binary": match_hamming}

    if precision not in matching_fn:
        raise ValueError(
            f"Unsupported precision type: {precision}. Must be one of {list(matching_fn.keys())}"
        )

    correct_at_k, _ = matching_fn[precision](
        desc,
        num_references=dataset.num_references,
        ground_truth=dataset.ground_truth,
        k_values=k_values,
    )
    return correct_at_k
