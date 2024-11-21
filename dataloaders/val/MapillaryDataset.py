import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset



class MSLS(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None):

        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "msls_val")
        self.dbImages = np.load("image_paths/msls_val_dbImages.npy")
        self.qImages = np.load("image_paths/msls_val_qImages.npy")
        self.ground_truth = np.load(
            "image_paths/msls_val_gt.npy", allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
