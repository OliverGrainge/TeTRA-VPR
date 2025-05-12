import pytorch_lightning as pl
from typing import List, Tuple, Callable
from models.transforms import get_transform
from dataloaders.test import TEST_DATASETS
from torch.utils.data import DataLoader
from dataloaders.utils.accuracy import get_recall_at_k
import torch
from PIL import Image
import numpy as np
from tabulate import tabulate


class Eval(pl.LightningModule):
    def __init__(
        self,
        model,
        test_set_names: List[str],
        test_dataset_dir: str,
        image_size: Tuple[int, int],
        num_workers: int,
        transform: Callable = None,
        batch_size: int = 1,
        k_values: List[int] = [1, 5, 10],
        matching_precision: str = "float32",
    ):
        super().__init__()
        self.model = model
        self.test_set_names = test_set_names
        self.test_dataset_dir = test_dataset_dir
        self.image_size = image_size
        self.num_workers = num_workers
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.matching_precision = matching_precision
        self.k_values = k_values
        self.transform = (
            self._default_transform(image_size) if transform is None else transform
        )
        self._check_test_set_names()

    def forward(self, x):
        return self.model(x)

    def _get_descriptor_dim(self):
        img = Image.fromarray(np.zeros((322, 322, 3), dtype=np.uint8))
        img = self.transform(img).unsqueeze(0)
        img = img.to(next(self.model.parameters()).device)
        desc = self.model(img)
        return desc.shape[1]

    def _default_transform(self, image_size: Tuple[int, int]):
        return get_transform(augmentation_level="None", image_size=image_size)

    def _check_test_set_names(self):
        for test_set_name in self.test_set_names:
            if test_set_name not in TEST_DATASETS:
                raise ValueError(
                    f"Invalid test set name: {test_set_name}, must be one of {TEST_DATASETS.keys()}"
                )

    def setup(self, stage: str):
        self.test_datasets = [
            TEST_DATASETS[val_set_name](
                val_dataset_dir=self.test_dataset_dir, input_transform=self.transform
            )
            for val_set_name in self.test_set_names
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            for test_ds in self.test_datasets
        ]

    def on_test_epoch_start(self):
        desc_dim = self._get_descriptor_dim()
        self.descriptors = {}
        for idx, test_dataset in enumerate(self.test_datasets):
            self.descriptors[self.test_set_names[idx]] = torch.zeros(
                len(test_dataset), desc_dim, dtype=torch.float16
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, idx = batch
        desc = self(imgs).detach().cpu().type(torch.float16)
        self.descriptors[self.test_set_names[dataloader_idx]][idx] = desc

    def on_test_epoch_end(self):
        table_data = []
        headers = [
            "Dataset",
            f"{self.matching_precision} R@1",
            f"{self.matching_precision} R@5",
            f"{self.matching_precision} R@10",
        ]

        for idx, test_set_name in enumerate(self.test_set_names):
            recall_at_k = get_recall_at_k(
                desc=self.descriptors[self.test_set_names[idx]],
                dataset=self.test_datasets[idx],
                k_values=self.k_values,
                precision=self.matching_precision,
            )
            table_data.append(
                [
                    test_set_name,
                    f"{recall_at_k[0]:.2f}",
                    f"{recall_at_k[1]:.2f}",
                    f"{recall_at_k[2]:.2f}",
                ]
            )
        self.results = table_data
        self.headers = headers
