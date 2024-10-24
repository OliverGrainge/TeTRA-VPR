import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.


config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
# Load the YAML configuration
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

DATASET_ROOT = os.path.join(config["Datasets"]["datasets_dir"], "sf_xl/")
GT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets/")
)
print(GT_ROOT)
print(GT_ROOT)
print(GT_ROOT)
print(GT_ROOT)
print(GT_ROOT)
print(GT_ROOT)

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(
        f"Please make sure the path {DATASET_ROOT} to Nordland dataset is correct"
    )


class SF_XL(Dataset):
    def __init__(self, which_ds="sf_xl_small_val", input_transform=None):

        assert which_ds.lower() in ["sf_xl_small_val", "sf_xl_small_test"]

        if "small" in which_ds:
            if "test" in which_ds:
                self.DATASET_ROOT = os.path.join(DATASET_ROOT, "small/test/")
            else:
                self.DATASET_ROOT = os.path.join(DATASET_ROOT, "small/val/")
        elif "processed" in which_ds:
            self.DATASET_ROOT = os.path.join(DATASET_ROOT, "processed/")
        else:
            raise ValueError(f"Dataset {which_ds} not found")

        self.input_transform = input_transform

        # Add error handling and more informative messages
        required_files = [
            f"{which_ds}_dbImages.npy",
            f"{which_ds}_qImages.npy",
            f"{which_ds}_gt.npy",
        ]

        for file in required_files:
            file_path = os.path.join(GT_ROOT, "SF_XL", file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        try:
            self.dbImages = np.load(
                os.path.join(GT_ROOT, "SF_XL", f"{which_ds}_dbImages.npy")
            )
            self.qImages = np.load(
                os.path.join(GT_ROOT, "SF_XL", f"{which_ds}_qImages.npy")
            )
            self.ground_truth = np.load(
                os.path.join(GT_ROOT, "SF_XL", f"{which_ds}_gt.npy"), allow_pickle=True
            )
        except Exception as e:
            print(f"Error loading dataset files: {e}")
            print(f"GT_ROOT: {GT_ROOT}")
            print(f"which_ds: {which_ds}")
            print(f"Current working directory: {os.getcwd()}")
            raise

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(self.DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
