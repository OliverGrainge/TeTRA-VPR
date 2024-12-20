import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class Tokyo247(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "tokyo247")
        self.which_set = which_set
        assert which_set == "test", "Tokyo247 only supports test set"
        # reference images names
        self.dbImages = np.load("dataloaders/val/image_paths/tokyo247_dbImages.npy")

        # query images names
        self.qImages = np.load("dataloaders/val/image_paths/tokyo247_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            "dataloaders/val/image_paths/tokyo247_gt.npy", allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index]))
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"Tokyo247_{self.which_set}"
