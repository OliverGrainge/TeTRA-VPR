import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the ESSEX dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.


class EssexDataset(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None, which_set="test"):

        assert which_set == "test", "EssexDataset only supports test set"
        self.input_transform = input_transform
        self.which_set = which_set
        self.dataset_root = os.path.join(val_dataset_dir, "ESSEX3IN1_dataset")

        # reference images names
        self.dbImages = np.load("dataloaders/val/image_paths/ESSEX_dbImages.npy")

        # query images names
        self.qImages = np.load("dataloaders/val/image_paths/ESSEX_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            "dataloaders/val/image_paths/ESSEX_gt.npy", allow_pickle=True
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
        return f"ESSEX3IN1_{self.which_set}"
