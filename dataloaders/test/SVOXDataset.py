import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class SVOX(Dataset):
    def __init__(
        self,
        val_dataset_dir=None,
        input_transform=None,
        condition=None,
        dataset_folder="svox",
    ):
        valid_conditions = ["overcast", "sun", "snow", "rain", "night", None]
        if condition not in valid_conditions:
            raise ValueError(
                f"SVOX only supports {', '.join(str(c) for c in valid_conditions[:-1])} and None. You gave {condition}"
            )

        self.condition = condition
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, dataset_folder)

        # Load dataset paths
        condition_str = f"_{condition}" if condition else ""
        base_path = "dataloaders/image_paths/svox"

        self.dbImages = np.load(f"{base_path}{condition_str}_test_dbImages.npy")
        self.qImages = np.load(f"{base_path}{condition_str}_test_qImages.npy")
        self.ground_truth = np.load(
            f"{base_path}{condition_str}_test_gt.npy",
            allow_pickle=True,
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index])).convert(
            "RGB"
        )
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        suffix = f"-{self.condition}" if self.condition else ""
        return f"SVOX{suffix}"


class SVOX_Night(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition="night")


class SVOX_Sun(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition="sun")


class SVOX_Rain(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition="rain")


class SVOX_Overcast(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition="overcast")


class SVOX_Snow(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition="snow")


class SVOX_None(SVOX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, condition=None)
