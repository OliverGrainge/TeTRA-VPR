import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import Dataset


class MSLS(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="val"):

        self.input_transform = input_transform
        self.which_set = which_set
        assert which_set in ["val", "test"], "MSLS only supports val and test set"
        self.dataset_root = os.path.join(val_dataset_dir)
        self.dbImages = np.load(f"dataloaders/val/image_paths/msls_{which_set}_dbImages.npy")
        self.qImages = np.load(f"dataloaders/val/image_paths/msls_{which_set}_qImages.npy")
        self.ground_truth = np.load(
            f"dataloaders/val/image_paths/msls_{which_set}_gt.npy", allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index])).convert("RGB")

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"MSLS_{self.which_set}"
