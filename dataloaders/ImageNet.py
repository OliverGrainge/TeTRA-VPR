import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from timm import create_model  # Using timm for Vision Transformer
from timm.data import create_transform
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from datasets import load_dataset

torch.set_float32_matmul_precision("medium")


class ImageNet(LightningModule):
    """
    PyTorch Lightning Model for training a Vision Transformer on ImageNet with Hugging Face Datasets.
    """

    def __init__(
        self,
        model,
        lr: float = 3e-4,  # Smaller learning rate for ViT
        weight_decay: float = 0.1,  # Higher weight decay for ViT
        batch_size: int = 32,
        workers: int = 4,
        warmup_epochs: int = 5,
        max_epochs: int = 90,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Define the transformations for training and validation
        self.train_transforms = create_transform(
            input_size=224,
            is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,  # Random Erasing probability
            re_mode="pixel",
            re_count=1,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # For validation, you can keep your existing transforms
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.fc = nn.Linear(model.descriptor_dim, 1000)

    def forward(self, x):

        return self.fc(self.model(x))

    def training_step(self, batch, batch_idx):

        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target, label_smoothing=0.1)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True
        )
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_val, on_step=True, on_epoch=True)
        self.log(f"{prefix}_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_acc5", acc5, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Compute total steps
        try:
            steps_per_epoch = len(self.train_dataloader())
        except TypeError:
            # If len is not available, estimate steps_per_epoch
            dataset_size = 1281167  # Number of images in ImageNet training set
            steps_per_epoch = dataset_size // self.batch_size
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        # Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def process_batch(self, batch, transform):
        images = []
        for b in batch:
            img = b["image"]
            # Convert images to RGB if they are not already in RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = transform(img)
            images.append(img)
        labels = [b["label"] for b in batch]
        return torch.stack(images), torch.tensor(labels)

    def train_dataloader(self):
        # Load the dataset using Hugging Face's datasets library (with streaming)
        train_dataset = load_dataset("imagenet-1k", split="train", streaming=False)

        # Apply transformations using a lambda function within the DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            collate_fn=lambda batch: self.process_batch(batch, self.train_transforms),
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader

    def val_dataloader(self):
        # Load the validation set

        val_dataset = load_dataset("imagenet-1k", split="validation", streaming=False)

        # Use the validation transformations
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
            collate_fn=lambda batch: self.process_batch(batch, self.val_transforms),
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
