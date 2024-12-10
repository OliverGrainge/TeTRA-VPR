import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

class SanFranciscoLarge(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="val"):
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "SF_XL")

        assert which_set in ["val", "test"]
        self.which_set = which_set
        # reference images names
        self.dbImages = np.load(f"dataloaders/val/image_paths/sf_xl_large_{which_set}_dbImages.npy")

        # query images names
        self.qImages = np.load(f"dataloaders/val/image_paths/sf_xl_large_{which_set}_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            f"dataloaders/val/image_paths/sf_xl_large_{which_set}_gt.npy", 
            allow_pickle=True
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
        return f"SFXL_large_{self.which_set}"


class SanFranciscoSmall(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="val"):
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "SF_XL")

        assert which_set in ["val", "test"]
        self.which_set = which_set
        # reference images names
        self.dbImages = np.load(f"dataloaders/val/image_paths/sf_xl_small_{which_set}_dbImages.npy")

        # query images names
        self.qImages = np.load(f"dataloaders/val/image_paths/sf_xl_small_{which_set}_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            f"dataloaders/val/image_paths/sf_xl_small_{which_set}_gt.npy", 
            allow_pickle=True
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
        return f"SFXL_small_{self.which_set}"