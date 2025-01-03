import glob
import os
import random
import tarfile
from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset


class TarImageDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        tar_paths = glob.glob(os.path.join(data_directory, "*.tar"))
        self.tar_paths = tar_paths
        self.transform = transform
        self.image_paths = []

        # Store tar file info and image paths for later access
        self.tar_info = []
        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                self.tar_info.extend(
                    [(tar_path, member) for member in members if member.isfile()]
                )

    def __len__(self):
        return len(self.tar_info)

    def __getitem__(self, idx):
        tar_path, member = self.tar_info[idx]
        with tarfile.open(tar_path, "r") as tar:
            file = tar.extractfile(member)
            image = Image.open(BytesIO(file.read()))
            image = image.convert("RGB")  # Convert to RGB if necessary

        width, height = image.size
        if width > height and width > 1024:
            height, height = 512, 512
            left = random.randint(0, width - height)
            right = left + height
            bottom = height
            image = image.crop((left, 0, right, bottom))

        if self.transform:
            image = self.transform(image)

        return image


class JPGDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        self.image_paths = []
        total_images = 0
        print(f"Scanning directory: {data_directory}")

        # Check if the directory contains subdirectories
        subdirs = [
            d
            for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, d))
        ]

        if subdirs:
            print("Found subdirectories. Scanning each:")
            for subdir in subdirs:
                subdir_path = os.path.join(data_directory, subdir)
                subdir_images = glob.glob(os.path.join(subdir_path, "*.jpg"))
                num_images = len(subdir_images)
                self.image_paths.extend(subdir_images)
                total_images += num_images
                print(f"  {subdir}: {num_images} images")
        else:
            print("No subdirectories found. Scanning for images in the main directory.")
            self.image_paths = glob.glob(os.path.join(data_directory, "*.jpg"))
            total_images = len(self.image_paths)
            print(f"  Main directory: {total_images} images")

        print(f"Total images found: {total_images}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx, attempts=0):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        try:
            image = image.convert("RGB")
        except (OSError, IOError) as e:
            print(f"Skipping corrupted image at index {idx}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx, attempts + 1)

        width, height = image.size
        if width > height and width > 2 * height:
            height, height = 512, 512
            left = random.randint(0, width - height)
            right = left + height
            bottom = height
            image = image.crop((left, 0, right, bottom))

        if self.transform:
            image = self.transform(image)

        return image


class DistillDataset(Dataset):
    def __init__(self, dataset, student_transform, teacher_transform):
        self.dataset = dataset
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        student_image = self.student_transform(image)
        teacher_image = self.teacher_transform(image)
        return student_image, teacher_image
