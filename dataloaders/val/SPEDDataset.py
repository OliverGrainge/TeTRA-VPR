import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the SPED dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.


class SPEDDataset(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None):
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "SPEDTEST")

        # reference images names
        self.dbImages = np.load("image_paths/SPED_dbImages.npy")

        # query images names
        self.qImages = np.load("image_paths/SPED_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            "image_paths/SPED_gt.npy", 
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
