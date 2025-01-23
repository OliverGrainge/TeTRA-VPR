import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb
from dataloaders.train.DistillDataset import DistillDataset, JPGDataset
from models.helper import get_model
from models.transforms import get_transform


class Distill(pl.LightningModule):
    """Knowledge distillation training module.

    This module implements teacher-student knowledge distillation using cosine and euclidean losses.
    """

    def __init__(
        self,
        student_model_backbone_arch: str,
        student_model_image_size: Tuple[int],
        train_dataset_dir: Tuple[str],
        lr: float,
        batch_size: int,
        weight_decay: float,
        image_size: Tuple[int],
        num_workers: int,
        augmentation_level: str,
        use_attn_loss: bool = False,
        token_loss_scale: float = 0.2,
        attn_loss_scale: float = 0.1,
    ):
        super().__init__()
        self.student_model_backbone_arch = student_model_backbone_arch
        self.student_model_image_size = student_model_image_size
        self.train_dataset_dir = train_dataset_dir
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.image_size = image_size
        self.num_workers = num_workers
        self.augmentation_level = augmentation_level
        self.use_attn_loss = use_attn_loss

        self.token_loss_scale = token_loss_scale
        self.attn_loss_scale = attn_loss_scale
        self._init_models()
        self._init_projections()

        # Log hyperparameters to wandb
        self.save_hyperparameters()

    def _init_models(self) -> None:
        """Initialize and configure teacher and student models."""
        self.teacher = self._setup_teacher()
        self.student = self._setup_student()
        self.freeze_model(self.teacher)
        self.student.set_qfactor(0.0)

    def _setup_teacher(self) -> nn.Module:
        teacher = get_model(
            preset="DinoV2_BoQ",
        )
        backbone = teacher.backbone
        return backbone

    def _setup_student(self) -> nn.Module:
        student = get_model(
            backbone_arch=self.student_model_backbone_arch,
            agg_arch="GeM",
            image_size=self.student_model_image_size,
        )
        backbone = student.backbone
        print("============================================= STUDENT =============================================")
        print(backbone)
        return backbone

    def _init_projections(self) -> None:
        """Initialize projection layers for feature alignment."""
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        teacher_transform = get_transform(preset="DinoV2_BoQ")
        student_transform = get_transform(
            augmentation_level=self.augmentation_level, image_size=self.image_size
        )
        teacher_img = teacher_transform(img)
        student_img = student_transform(img)
        teacher_features = self.teacher.forward_distill(
            teacher_img[None, ...].to(next(self.student.parameters()).device)
        )
        student_features = self.student.forward_distill(
            student_img[None, ...].to(next(self.student.parameters()).device)
        )

        if teacher_features.shape != student_features.shape:
            assert (
                teacher_features.shape[-1] > student_features.shape[-1]
            ), "Teacher features must have more than or equal dimensions than student features. Teacher: {teacher_features.shape}, Student: {student_features.shape}"
            self.student_projection = nn.Linear(
                student_features.shape[-1], teacher_features.shape[-1]
            )
            self.teacher_projection = nn.Identity()
            print("--------------------------------")
            print(
                f"Expanding student features to match teacher features from {student_features.shape} to {teacher_features.shape}"
            )
            print("--------------------------------")
        else:
            self.student_projection = nn.Identity()
            self.teacher_projection = nn.Identity()

    def setup(self, stage=None):
        if stage == "fit":
            wandb.define_metric(f"cosine_loss", summary="min")
            wandb.define_metric(f"euclidian_loss", summary="min")
            wandb.define_metric(f"train_loss", summary="min")

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _progressive_quant_scheduler(self):
        x = (
            (
                self.global_step
                / (
                    self.trainer.estimated_stepping_batches
                    // self.trainer.accumulate_grad_batches
                )
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
            num_training_steps=self.trainer.estimated_stepping_batches
            // self.trainer.accumulate_grad_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        wandb.summary
        return [optimizer], [scheduler_config]

    def _compute_features(
        self, student_images, teacher_images, use_attn_loss: bool = False
    ):
        B = student_images.shape[0]
        if use_attn_loss:
            with torch.no_grad():
                teacher_features, teacher_attn = self.teacher.forward_distill(
                    teacher_images, return_attn=True
                )
            student_features, student_attn = self.student.forward_distill(
                student_images, return_attn=True
            )
        else:
            with torch.no_grad():
                teacher_features = self.teacher.forward_distill(teacher_images)
            student_features = self.student.forward_distill(student_images)
        teacher_features = self.teacher_projection(teacher_features)
        student_features = self.student_projection(student_features)

        assert (
            teacher_features.shape == student_features.shape
        ), "teacher and student features must have the same shape"
        if use_attn_loss:
            return student_features, teacher_features, student_attn, teacher_attn
        else:
            return student_features, teacher_features

    @staticmethod
    def _compute_mse_loss(student_features, teacher_features):
        # Normalize the embeddings
        student_normalized = F.normalize(student_features, p=2, dim=-1)
        teacher_normalized = F.normalize(teacher_features, p=2, dim=-1)

        return (
            F.mse_loss(student_normalized, teacher_normalized, reduction="mean") * 100
        )

    @staticmethod
    def _compute_attn_loss(student_attn, teacher_attn, temperature=2.0, scale=50.0):
        assert (
            student_attn.shape == teacher_attn.shape
        ), "Student and teacher attention maps must have the same shape"

        # Reshape to (batch * n_layers , n_tokens, n_tokens)
        B, L, H, N, N = student_attn.shape
        student_attn = student_attn.reshape(-1, N, N)
        teacher_attn = teacher_attn.reshape(-1, N, N)

        # Apply temperature scaling and compute distributions
        teacher_dist = torch.softmax(teacher_attn / temperature, dim=-1)

        # KLDivLoss expects log probabilities for the student (first argument)
        loss = torch.nn.KLDivLoss(reduction="mean", log_target=False)(
            torch.log_softmax(
                student_attn / temperature, dim=-1
            ),  # student log probabilities
            teacher_dist,  # teacher probabilities (not log)
        )

        return (
            loss * (temperature**2) * scale
        )  # Scale the loss as per the temperature scaling paper

    def training_step(self, batch, batch_idx):
        # compute features
        student_images, teacher_images = batch
        if self.use_attn_loss:
            student_features, teacher_features, student_attn, teacher_attn = (
                self._compute_features(
                    student_images, teacher_images, use_attn_loss=True
                )
            )
        else:
            student_features, teacher_features = self._compute_features(
                student_images, teacher_images, use_attn_loss=False
            )

        # compute losses
        cls_loss = self._compute_mse_loss(
            teacher_features[:, 0, :], student_features[:, 0, :]
        )
        token_loss = (
            self._compute_mse_loss(
                teacher_features[:, 1:, :], student_features[:, 1:, :]
            )
            * self.token_loss_scale
        )
        if self.use_attn_loss:
            attn_loss = (
                self._compute_attn_loss(student_attn, teacher_attn)
                * self.attn_loss_scale
            )
        else:
            attn_loss = torch.tensor(0.0).to(self.device)

        # progressive quantization
        qfactor = self._progressive_quant_scheduler()
        self.student.set_qfactor(qfactor)
        self.log("qfactor", qfactor, on_step=True)

        total_loss = cls_loss + token_loss + attn_loss

        self.log("cls_loss", cls_loss, on_step=True)
        self.log("token_loss", token_loss, on_step=True)
        self.log("attn_loss", attn_loss, on_step=True)
        self.log("train_loss", total_loss, on_step=True, prog_bar=True)
        return total_loss

    def train_dataloader(self):
        for path in self.train_dataset_dir:
            if not os.path.isdir(path):
                raise ValueError(f"Invalid data directory: {path}")

        train_dataset = JPGDataset(self.train_dataset_dir)
        student_transform = get_transform(
            augmentation_level=self.augmentation_level, image_size=self.image_size
        )
        teacher_transform = get_transform(preset="DinoV2_BoQ")
        print("============================================= TRANSFORMS =============================================")
        print("Student transform: ", student_transform)
        print("Teacher transform: ", teacher_transform)
        print("======================================================================================================")
        dataset = DistillDataset(
            dataset=train_dataset,
            student_transform=student_transform,
            teacher_transform=teacher_transform,
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
