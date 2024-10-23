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
from collections import defaultdict
from matching.global_cosine_sim import global_cosine_sim
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
    return features["global_desc"].shape[1]


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
        self.image_paths = []
        total_images = 0
        print(f"Scanning directory: {data_directory}")

        # Check if the directory contains subdirectories
        subdirs = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]

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
        matching_function=global_cosine_sim,
    ):
        super().__init__()
        self.config = config
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.lr = config["lr"]
        self.data_directory = config["data_directory"]
        self.teacher_preset = args.teacher_preset
        self.val_set_names = args.val_set_names
        self.matching_function = global_cosine_sim
        self.backbone_arch = student_backbone_arch

        self.teacher = get_model(preset=args.teacher_preset)
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            out_dim=get_feature_dim(self.teacher, self.teacher_train_transform()),
        )

        freeze_model(self.teacher)

    def setup(self, stage=None):
            # Setup for 'fit' or 'validate'self
            if stage == "fit" or stage == "validate":
                self.val_datasets = []
                for val_set_name in self.val_set_names:
                    if "pitts30k" in val_set_name.lower():
                        from dataloaders.val.PittsburghDataset import PittsburghDataset

                        self.val_datasets.append(
                            PittsburghDataset(
                                which_ds=val_set_name, input_transform=self.student_val_transform()
                            )
                        )
                    elif val_set_name.lower() == "msls_val":
                        from dataloaders.val.MapillaryDataset import MSLS

                        self.val_datasets.append(MSLS(input_transform=self.student_val_transform()))
                    elif val_set_name.lower() == "nordland":
                        from dataloaders.val.NordlandDataset import NordlandDataset

                        self.val_datasets.append(
                            NordlandDataset(input_transform=self.transform)
                        )
                    elif val_set_name.lower() == "sped":
                        from dataloaders.val.SPEDDataset import SPEDDataset

                        self.val_datasets.append(
                            SPEDDataset(input_transform=self.student_val_transform())
                        )
                    elif "sf_xl" in val_set_name.lower() and "val" in val_set_name.lower() and "small" in val_set_name.lower():
                        from dataloaders.val.SF_XL import SF_XL

                        self.val_datasets.append(
                            SF_XL(which_ds="sf_xl_small_val", input_transform=self.student_val_transform())
                        )
                    elif "sf_xl" in val_set_name.lower() and "test" in val_set_name.lower() and "small" in val_set_name.lower():
                        from dataloaders.val.SF_XL import SF_XL

                        self.val_datasets.append(
                            SF_XL(which_ds="sf_xl_small_test", input_transform=self.student_val_transform())
                        )
                    else:
                        raise NotImplementedError(
                            f"Validation set {val_set_name} not implemented"
                        )
        

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
        # Apply weight decay only to weight parameters, not biases or normalization layers
        decay_params = []
        no_decay_params = []
        for name, param in self.student.named_parameters():
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Use weight decay for ViT models
        weight_decay = 0.05 if 'vit' in self.backbone_arch.lower() else 0.0

        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.lr)

        # Calculate total steps for warmup and cosine annealing
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)  # 10% of total steps for warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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
    
    def student_val_transform(self):
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
    
    def teacher_val_transform(self):
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
    
    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    shuffle=False,
                    num_workers=self.num_workers,
                    batch_size=self.batch_size,
                )
            )
        return val_dataloaders
    

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = {}
        for name in self.val_set_names:
            self.validation_outputs[name] = defaultdict(list)

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(value.detach().cpu())
        return descriptors["global_desc"].detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""

        results_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets): 
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _ = self.matching_function(**set_outputs, num_references=val_dataset.num_references, ground_truth=val_dataset.ground_truth)
            self.log_dict(
                recalls_dict,
                prog_bar=True,
                logger=True,
            )
            results_dict[val_set_name] = recalls_dict

            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            for metric, value in recalls_dict.items():
                table.add_row([metric, f"{value:.4f}"])
            
            print(f"\nResults for {val_set_name}:")
            print(table)
        
        return results_dict

