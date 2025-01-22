import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb
from dataloaders.train.DistillDataset import DistillDataset, JPGDataset
from models.helper import get_model
from models.transforms import get_transform


class Distill(pl.LightningModule):
    def __init__(
        self,
        student_model_backbone_arch,
        student_model_agg_arch,
        student_model_image_size,
        teacher_model_preset,
        train_dataset_dir,
        augmentation_level="Moderate",
        use_attention=True,
        use_progressive_quant=True,
        weight_decay=0.01,
        batch_size=32,
        num_workers=4,
        image_size=224,
        lr=1e-3,
        mse_loss_mult=2,
        latent_dim=512,
    ):
        super().__init__()
        # Model-related attributes
        self.teacher_model_preset = teacher_model_preset
        self.teacher = get_model(
            preset=teacher_model_preset,
            normalize=False,
        )
        self.student = get_model(
            backbone_arch=student_model_backbone_arch,
            agg_arch=student_model_agg_arch,
            image_size=student_model_image_size,
            normalize=False,
        )
        self.student = self.student.to("cuda")
        self.teacher = self.teacher.to("cuda")

        # Dataset and data-related attributes
        self.train_dataset_dir = train_dataset_dir
        self.augmentation_level = augmentation_level
        self.image_size = image_size

        # Training-related attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.mse_loss_mult = mse_loss_mult
        self.use_attention = use_attention
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim

        # progressive quantization
        self.use_progressive_quant = use_progressive_quant
        if use_progressive_quant and not hasattr(self.student.backbone, "set_qfactor"):
            raise Exception(
                "Student backbone does not have support pregressive quant method"
            )
        if use_progressive_quant:
            self.student.backbone.set_qfactor(0.0)
        else:
            if hasattr(self.student.backbone, "set_qfactor"):
                self.student.backbone.set_qfactor(1.0)

        self.freeze_model(self.teacher)
        print(repr(self.student))

        self._setup_projection()    

    def freeze_model(self, model): 
        for param in model.parameters(): 
            param.requires_grad=False

    def _setup_projection(self): 
        img = torch.randn(1, 3, *self.image_size)
        out = self.student(img.to(next(self.student.parameters()).device))
        feature_dim = out.shape[-1]
        self.teacher_projection = nn.Sequential(nn.Linear(feature_dim, self.latent_dim), nn.LayerNorm(self.latent_dim))
        self.student_projection = nn.Sequential(nn.Linear(feature_dim, self.latent_dim), nn.LayerNorm(self.latent_dim))


    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            wandb.define_metric(f"cosine_loss", summary="min")
            wandb.define_metric(f"euclidian_loss", summary="min")
            wandb.define_metric(f"total_loss", summary="min")

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
        return [optimizer], [scheduler_config]

    def _compute_features(self, student_images, teacher_images):
        B = student_images.shape[0]
        teacher_features = self.teacher_projection(self.teacher(teacher_images))
        student_features = self.student_projection(self(student_images))


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
        if self.use_progressive_quant:
            qfactor = self._progressive_quant_scheduler()
            self.student.backbone.set_qfactor(qfactor)
            self.log("qfactor", qfactor, on_step=True)

        if (
            hasattr(self.student.backbone, "set_qfactor")
            and not self.use_progressive_quant
        ):
            for module in self.student.backbone.modules():
                if hasattr(module, "qfactor"):
                    self.log("qfactor", module.qfactor, on_step=True)
                    break

        total_loss = euc_loss + cos_loss

        #print("== total_loss", total_loss.item(), "cos_loss", cos_loss.item(), "euc_loss", euc_loss.item())

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
