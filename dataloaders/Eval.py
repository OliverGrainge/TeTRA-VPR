from pytorch_lightning import LightningDataModule
from typing import List, Tuple, Callable
from models.transforms import get_transform
from dataloaders.test import TEST_DATASETS
from torch.utils.data import DataLoader
from dataloaders.utils.accuracy import compute_recall_at_k
import torch
from PIL import Image 
import numpy as np

class Eval(LightningDataModule):
    def __init__(self, model, test_set_names: List[str], test_dataset_dir: str, image_size: Tuple[int, int], num_workers: int, transform: Callable=None, batch_size: int=1, k_values: List[int]=[1, 5, 10], precision: str="float32"):
        super().__init__()
        self.model = model
        self.test_set_names = test_set_names
        self.test_dataset_dir = test_dataset_dir
        self.image_size = image_size
        self.num_workers = num_workers
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self._default_transform(image_size) if transform is None else transform
        self._check_test_set_names()

    def forward(self, x): 
        return self.model(x)

    def _get_descriptor_dim(self): 
        img = Image.fromarray(np.zeros((3, 322, 322), dtype=np.uint8))
        desc = self.model(self.transform(img).unsqueeze(0).to(next(self.model.parameters()).device))
        return desc.shape[1]
    
    def _default_transform(self, image_size: Tuple[int, int]):
        return get_transform(augmentation_level="None", image_size=image_size)

    def _check_test_set_names(self): 
        for test_set_name in self.test_set_names:
            if test_set_name not in TEST_DATASETS:
                raise ValueError(f"Invalid test set name: {test_set_name}")

    def setup(self, stage: str):
        self.test_datasets = [TEST_DATASETS[val_set_name](self.val_dataset_dir, self.image_size, self.num_workers, self.transform) for val_set_name in self.val_set_names]

    def test_dataloader(self):
        return [DataLoader(self.val_datasets, batch_size=self.batch_size, num_workers=self.num_workers) for val_set_name in self.val_set_names]
    
    def on_test_epoch_start(self): 
        desc_dim = self._get_descriptor_dim()
        self.descriptors = {} 
        for idx, test_dataset in enumerate(self.test_datasets): 
            self.descriptors[self.test_set_names[idx]] = torch.zeros(test_dataset.num_images, desc_dim, dtype=torch.float16)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, idx = batch 
        desc = self(imgs).detach().cpu().type(torch.float16)
        self.descriptors[self.test_set_names[dataloader_idx]][idx] = desc

    def on_test_epoch_end(self): 
        for test_set_name in self.test_set_names:
            recall_at_k = compute_recall_at_k(self.descriptors[test_set_name], self.test_datasets[test_set_name], [1, 5, 10])

