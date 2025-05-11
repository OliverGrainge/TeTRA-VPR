import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MSLS(Dataset):
    """Dataset class for Mapillary Street Level Sequences visual place recognition benchmark"""
    
    def __init__(self, val_dataset_dir=None, input_transform=None, dataset_folder="msls"):
        """Initialize the dataset
        
        Args:
            val_dataset_dir: Root directory containing dataset images
            input_transform: Optional transforms to apply to images
        """
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, dataset_folder)
        
        # Load image paths and ground truth
        self.dbImages = np.load("dataloaders/image_paths/msls_test_dbImages.npy")
        self.qImages = np.load("dataloaders/image_paths/msls_test_qImages.npy") 
        self.ground_truth = np.load("dataloaders/image_paths/msls_test_gt.npy",
                                   allow_pickle=True)

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
        img_path = os.path.join(self.dataset_root, self.images[index])
        img = Image.open(img_path).convert("RGB")
        
        if self.input_transform:
            img = self.input_transform(img)
            
        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return "MSLS"
