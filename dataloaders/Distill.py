import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

import wandb
from dataloaders.train.DistillDataset import DistillDataset, JPGDataset
from models.helper import get_model
from models.transforms import get_transform
import wandb


class Distill(pl.LightningModule):
    """Knowledge distillation training module.
    
    This module implements teacher-student knowledge distillation using cosine and euclidean losses.
    """
    
    def __init__(self, teacher_model_preset: str, student_model_backbone_arch: str, student_model_agg_arch: str, student_model_image_size: Tuple[int], train_dataset_dir: Tuple[str], lr: float, batch_size: int, weight_decay: float, image_size: Tuple[int], num_workers: int, augmentation_level: str, latent_dim: int=512, mse_loss_mult: float=2.0):
        super().__init__()
        self.teacher_model_preset = teacher_model_preset
        self.student_model_backbone_arch = student_model_backbone_arch
        self.student_model_agg_arch = student_model_agg_arch
        self.student_model_image_size = student_model_image_size
        self.train_dataset_dir = train_dataset_dir
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.mse_loss_mult = mse_loss_mult
        self.image_size = image_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level
        self.latent_dim = latent_dim
        self._init_models()
        self._init_projections()
        
        # Log hyperparameters to wandb
        self.save_hyperparameters()
        
    def _init_models(self) -> None:
        """Initialize and configure teacher and student models."""
        self.teacher = self._setup_teacher()
        self.student = self._setup_student()
        self.freeze_model(self.teacher)
        self.student.backbone.set_qfactor(0.0)
        
    def _setup_teacher(self) -> nn.Module:
        teacher = get_model(
            preset=self.teacher_model_preset,
            normalize=False,
        )
        return teacher
        
    def _setup_student(self) -> nn.Module:
        student = get_model(
            backbone_arch=self.student_model_backbone_arch,
            agg_arch=self.student_model_agg_arch,
            image_size=self.student_model_image_size,
            normalize=False,
        )
        return student

    def _init_projections(self) -> None:
        """Initialize projection layers for feature alignment."""
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        teacher_transform = get_transform(preset=self.teacher_model_preset)
        student_transform = get_transform(augmentation_level=self.augmentation_level, image_size=self.image_size)
        teacher_img = teacher_transform(img)
        student_img = student_transform(img)
        teacher_features = self.teacher(teacher_img[None, ...].to(next(self.student.parameters()).device))
        student_features = self.student(student_img[None, ...].to(next(self.student.parameters()).device))

        teacher_feature_dim = teacher_features.shape[-1]
        student_feature_dim = student_features.shape[-1]
        
        self.teacher_projection = nn.Sequential(
            nn.Linear(teacher_feature_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )
        
        self.student_projection = nn.Sequential(
            nn.Linear(student_feature_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )


    def setup(self, stage=None):
        if stage == "fit":
            wandb.define_metric(f"cosine_loss", summary="min")
            wandb.define_metric(f"euclidian_loss", summary="min")
            wandb.define_metric(f"train_loss", summary="min")

    def freeze_model(self, model): 
        for param in model.parameters(): 
            param.requires_grad=False

    def _progressive_quant_scheduler(self):
        x = (
            (
                self.global_step
                / (self.trainer.estimated_stepping_batches // self.trainer.accumulate_grad_batches)
            )
            * 12
        ) - 6
        qfactor = 1 / (1 + math.exp(-x))
        return qfactor

    def forward(self, x):
        return self.student(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.student.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Calculate the total number of training steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches // self.trainer.accumulate_grad_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        wandb.summary
        return [optimizer], [scheduler_config]

    def _compute_features(self, student_images, teacher_images):
        B = student_images.shape[0]
        teacher_features = self.teacher(teacher_images)
        student_features = self(student_images)

        teacher_features = self.teacher_projection(teacher_features)
        student_features = self.student_projection(student_features)


        assert (
            teacher_features.shape == student_features.shape
        ), "teacher and student features must have the same shape"

        return student_features, teacher_features

    @staticmethod
    def _compute_cosine_loss(student_features, teacher_features):
        student_features = F.normalize(student_features, p=2, dim=-1)
        teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        cosine_loss = 1 - F.cosine_similarity(teacher_features, student_features)
        return cosine_loss.mean()

    @staticmethod
    def _compute_euclidian_loss(student_features, teacher_features):
        return F.mse_loss(teacher_features, student_features, reduction='mean')

    def training_step(self, batch, batch_idx):
        # compute features
        student_images, teacher_images = batch
        student_features, teacher_features = self._compute_features(
            student_images, teacher_images
        )


        # compute losses
        euc_loss = self.mse_loss_mult * self._compute_euclidian_loss(
            teacher_features, student_features
        )
        cos_loss = self._compute_cosine_loss(teacher_features, student_features)

        # progressive quantization
        qfactor = self._progressive_quant_scheduler()
        self.student.backbone.set_qfactor(qfactor)
        self.log("qfactor", qfactor, on_step=True)


        total_loss = euc_loss + cos_loss

        self.log("cosine_loss", cos_loss, on_step=True)
        self.log("euclidian_loss", euc_loss, on_step=True)
        self.log("train_loss", total_loss, on_step=True, prog_bar=True)
        return total_loss

    def train_dataloader(self):
        for path in self.train_dataset_dir:
            if not os.path.isdir(path):
                raise ValueError(f"Invalid data directory: {path}")

        train_dataset = JPGDataset(self.train_dataset_dir)
        dataset = DistillDataset(
            dataset=train_dataset,
            student_transform=get_transform(
                augmentation_level=self.augmentation_level, image_size=self.image_size
            ),
            teacher_transform=get_transform(preset=self.teacher_model_preset),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def state_dict(self):
        sd = self.student.state_dict()
        return sd
