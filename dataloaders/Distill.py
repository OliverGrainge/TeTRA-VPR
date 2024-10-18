import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import pytorch_lightning as pl
from pytorch_metric_learning import miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import tarfile
from io import BytesIO
import json
import glob
import random
import os

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

from models.helper import get_model, get_transform

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_feature_dim(model, transform):
    x = torch.randint(0, 255, size=(3, 512, 512), dtype=torch.uint8)
    x_np = x.numpy()
    x_img = Image.fromarray(x_np.transpose(1, 2, 0))
    x_transformed = transform(x_img)
    features = model(x_transformed[None, :].to(next(model.parameters()).device))
    return features.shape[1]


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


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
        if width > height:
            height, height = 512, 512
            left = random.randint(0, width - height)
            right = left + height
            bottom = height
            image = image.crop((left, 0, right, bottom))

        if self.transform:
            image = self.transform(image)

        return image


class ImageDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        self.image_paths = glob.glob(os.path.join(data_directory, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")

        width, height = image.size
        if width > height:
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


class VPRDistill(pl.LightningModule):
    def __init__(
        self,
        config,
        args,
        student_backbone_arch="vit_small",
        student_agg_arch="cls",
    ):
        super().__init__()
        self.config = config
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.lr = config["lr"]
        self.data_directory = config["data_directory"]
        self.teacher_preset = args.teacher_preset

        self.teacher = get_model(preset=args.teacher_preset)
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            out_dim=get_feature_dim(self.teacher, self.teacher_train_transform()),
        )

        freeze_model(self.teacher)

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        student_images, teacher_images = batch
        teacher_features = self.teacher(teacher_images)
        student_features = self(student_images)
        teacher_features = F.normalize(teacher_features["global_desc"], dim=1)
        student_features = F.normalize(student_features["global_desc"], dim=1)
        mse_loss = F.mse_loss(teacher_features, student_features)
        cosine_loss = (
            1 - F.cosine_similarity(teacher_features, student_features)
        ).mean()
        loss = 1000 * mse_loss + cosine_loss
        self.log("train_loss", loss)
        self.log("mse_loss", mse_loss)
        self.log("cosine_loss", cosine_loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr)
        return optimizer

    def student_train_transform(self):
        return T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )

    def teacher_train_transform(self):
        return get_transform(self.teacher_preset)

    def train_dataloader(self):
        paths = os.listdir(self.data_directory)
        if not os.path.isdir(self.data_directory):
            raise ValueError(f"Invalid data directory: {self.data_directory}")

        if "*tar" in paths[0]:
            train_dataset = TarImageDataset(self.data_directory)
        else:
            train_dataset = ImageDataset(self.data_directory)

        dataset = DistillDataset(
            train_dataset,
            self.student_train_transform(),
            self.teacher_train_transform(),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
