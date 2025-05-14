import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SFXLOcclusion(Dataset):
    """Dataset class for Pittsburgh 30k visual place recognition benchmark"""

    def __init__(
        self,
        val_dataset_dir=None,
        input_transform=None,
        dataset_folder="sf_xl",
    ):
        """Initialize the dataset

        Args:
            val_dataset_dir: Root directory containing the dataset
            input_transform: Optional transforms to apply to images
            dataset_folder: Name of the dataset folder
        """
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, dataset_folder)

        # Load image paths and ground truth
        self.dbImages = np.load(
            "dataloaders/image_paths/sf_xl_occlusion_test_dbImages.npy",
            allow_pickle=True,
        )
        self.qImages = np.load(
            "dataloaders/image_paths/sf_xl_occlusion_test_qImages.npy",
            allow_pickle=True,
        )
        self.ground_truth = np.load(
            "dataloaders/image_paths/sf_xl_occlusion_test_gt.npy", allow_pickle=True
        )

        # Combine reference and query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        """Get an image and its index

        Args:
            index: Index of image to retrieve

        Returns:
            tuple: (image, index)
        """
        img_path = os.path.join(self.dataset_root, self.images[index].lstrip("/"))
        img = Image.open(img_path)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return "SFXL-Occlusion"
