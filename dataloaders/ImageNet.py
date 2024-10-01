import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
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

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


class QLambdaScheduler: 
    def __init__(self, model, max_steps, init_val=0.0): 
        self.layers = []
        self.max_steps = max_steps 

        for module in model: 
            if isinstance(module, nn.Linear) and hasattr(module, 'q_lamda'): 
                module.q_lambda = torch.tensor(init_val)
                self.layers.append(module)
        self.current_step = 0 

    def step(self):
        t = self.current_step / self.max_steps
        t = torch.tensor((t * 10) - 4)  # Scale t for sigmoid function
        val = 1 / (1 + torch.exp(-t))  # Sigmoid decay
        # Update q_lambda for each module
        for module in self.layers:
            module.q_lambda = torch.tensor(val)  # Update q_lambda attribute
        self.current_step += 1

    def get_val(self):
        t = self.current_step / self.max_steps
        t = torch.tensor((t * 10) - 4)  # Scale t for sigmoid function
        val = 1 / (1 + torch.exp(-t))
        return val

class QRegScheduler: 
    def __init__(self, max_steps, scale): 
        self.max_steps = max_steps 
        self.scale = scale 
        self.current_step = 0

    def step(self): 
        self.current_step += 1 

    def get_scalar(self):
        t = self.current_step/self.max_steps 
        return self.scale


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
        warmup_epochs: int = 3,
        max_epochs: int = 90,
        opt_type: str = "bitnet",
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
        self.max_steps = max_epochs * len(self.train_dataloader())
        self.opt_type = opt_type
        self.weight_decay_reset_epoch = int(max_epochs * 0.5)
        self.quantization_scheduler = QLambdaScheduler(model, self.max_steps)
        self.reg_scheduler = QRegScheduler(self.max_steps, 0.000001)
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


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target, label_smoothing=0.1)
        loss_reg = self.reg_loss()
        alpha = self.reg_scheduler.get_scalar()
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        print(loss_train.item(), alpha*loss_reg)
        loss = loss_train + alpha * loss_reg
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("reg_loss", alpha * loss_reg, on_step=True, on_epoch=True, logger=True)
        self.log("loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True
        )
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def reg_loss(self, x): 
        reg = torch.tensor(0.0)
        for module in self.model.modules(): 
            if isinstance(module, nn.Linear) and hasattr(module, 'compute_reg'):
                reg += module.compute_reg()
        return reg

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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank == 0: 
            self.quantization_scheduler.step()
            self.log(self.quantization_scheduler.get_val())

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

    def on_train_epoch_start(self):
        """
        Hook called at the start of each epoch. Logs the current weight_decay and resets it if the current epoch matches the reset epoch.
        """
        optimizer = self.optimizers()  # Get the optimizer (not iterable)

        start_descent = int(0.3 * self.max_epochs)
        end_descent = int(0.5 * self.max_epochs)
        delta = 0.1 / (end_descent - start_descent)
        # Iterate through parameter groups in the single optimizer
        for param_group in optimizer.param_groups:
            # Log the current weight decay before any modification
            current_weight_decay = param_group.get("weight_decay", 0.0)
            self.log("weight_decay", current_weight_decay, prog_bar=True, logger=True)

            # If the current epoch matches the reset epoch and using bitnet, reset weight decay
            if (
                self.current_epoch > start_descent
                and self.current_epoch < end_descent
                and self.opt_type == "bitnet"
            ):
                param_group["weight_decay"] = param_group["weight_decay"] - delta
                param_group["weight_decay"] = max(param_group["weight_decay"], 0)
                self.log(
                    "weight_decay",
                    param_group["weight_decay"],
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Compute total steps
        steps_per_epoch = len(self.train_dataloader())
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        # raise Exception
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
        train_dataset = load_dataset(
            "imagenet-1k", split=f"train", cache_dir=config["Datasets"]["datasets_dir"]
        )

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
        val_dataset = load_dataset(
            "imagenet-1k",
            split="validation",
            cache_dir=config["Datasets"]["datasets_dir"],
        )

        # Use the validation transformations
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            collate_fn=lambda batch: self.process_batch(batch, self.val_transforms),
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")
