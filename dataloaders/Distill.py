import math
import os
import sys
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb
from dataloaders.train.DistillDataset import DistillDataset, JPGDataset, TarImageDataset
from dataloaders.utils.Distill.attention import get_attn, remove_hooks
from dataloaders.utils.Distill.funcs import L2Norm, freeze_model, get_feature_dim
from dataloaders.utils.Distill.schedulers import QuantScheduler, WeightDecayScheduler
from matching.match_cosine import match_cosine
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
        val_dataset_dir,
        augmentation_level="Moderate",
        use_attention=True,
        use_progressive_quant=True,
        weight_decay=0.01,
        batch_size=32,
        num_workers=4,
        image_size=224,
        lr=1e-3,
        mse_loss_mult=200,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
        # Model-related attributes
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
        self.val_dataset_dir = val_dataset_dir
        self.val_set_names = val_set_names
        self.augmentation_level = augmentation_level
        self.image_size = image_size

        # Training-related attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.mse_loss_mult = mse_loss_mult
        self.use_attention = use_attention
        self.weight_decay = weight_decay

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

        freeze_model(self.teacher)
        print(repr(self.student))

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            val_transform = get_transform(
                augmentation_level="None", image_size=self.image_size
            )
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset30k

                    self.val_datasets.append(
                        PittsburghDataset30k(
                            val_dataset_dir=self.val_dataset_dir,
                            input_transform=val_transform,
                            which_set="val",
                        )
                    )
                elif "msls" in val_set_name.lower():
                    from dataloaders.val.MapillaryDataset import MSLS

                    self.val_datasets.append(
                        MSLS(
                            val_dataset_dir=self.val_dataset_dir,
                            input_transform=val_transform,
                            which_set="val",
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )
            for val_set_name in self.val_set_names:
                wandb.define_metric(f"{val_set_name}_R1", summary="max")
            wandb.define_metric(f"Cosine Loss", summary="min")
            wandb.define_metric(f"Euclidian Loss", summary="min")
            wandb.define_metric(f"Total Loss", summary="min")
            self.num_training_steps = self.trainer.max_epochs * len(
                self.train_dataloader()
            )

    def _progressive_quant_scheduler(self):
        x = (
            (
                self.global_step
                / (self.num_training_steps // self.trainer.accumulate_grad_batches)
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
        num_training_steps = self.trainer.max_epochs * len(self.train_dataloader())
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
            // self.trainer.accumulate_grad_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]

    def _compute_features(self, student_images, teacher_images):
        B = student_images.shape[0]
        teacher_features = self.teacher(teacher_images)
        student_features = self(student_images)

        assert (
            teacher_features.shape == student_features.shape
        ), "teacher and student features must have the same shape"

        return student_features, teacher_features

    @staticmethod
    def _compute_cosine_loss(student_features, teacher_features):
        student_features = F.normalize(student_features, p=2, dim=-1)
        teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        print("============================================== teacher features post norm ", teacher_features.norm(dim=1)[:5])
        print("============================================== student_features post norm ", student_features.norm(dim=1)[:5])
        cosine_loss = 1 - F.cosine_similarity(teacher_features, student_features)
        return cosine_loss.mean()

    @staticmethod
    def _compute_euclidian_loss(student_features, teacher_features):
        return F.mse_loss(teacher_features, student_features)

    def training_step(self, batch, batch_idx):
        # compute features
        student_images, teacher_images = batch
        student_features, teacher_features = self._compute_features(
            student_images, teacher_images
        )

        print(
            "============================================== teacher features pre norm ",
            teacher_features.norm(dim=1)[:5],
        )
        print(
            "============================================== student_features pre norm ",
            student_features.norm(dim=1)[:5],
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

        print("== total_loss", total_loss.item(), "cos_loss", cos_loss.item(), "euc_loss", euc_loss.item())

        self.log("Cosine Loss", cos_loss, on_step=True)
        self.log("Euclidian Loss", euc_loss, on_step=True)
        self.log("Total Loss", total_loss, on_step=True, prog_bar=True)
        return total_loss

    def train_dataloader(self):
        if not os.path.isdir(self.train_dataset_dir):
            raise ValueError(f"Invalid data directory: {self.train_dataset_dir}")

        train_dataset = JPGDataset(self.train_dataset_dir)
        print("============================================== agumentation_level ", self.augmentation_level)
        raise Exception("stop")
        dataset = DistillDataset(
            dataset=train_dataset,
            student_transform=get_transform(
                augmentation_level=self.augmentation_level, image_size=self.image_size
            ),
            teacher_transform=get_transform(preset="DinoSalad"),
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
        self.validation_outputs = {}
        for name in self.val_set_names:
            self.validation_outputs[name] = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        descriptors = self(places)

        self.validation_outputs[self.val_set_names[dataloader_idx]].append(
            descriptors.detach().cpu()
        )
        return descriptors.detach().cpu()

    def on_validation_epoch_end(self):
        full_recalls_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            descriptors = torch.concat(set_outputs, dim=0)

            recalls_dict, _, _ = match_cosine(
                descriptors,
                num_references=val_dataset.num_references,
                ground_truth=val_dataset.ground_truth,
            )

            for k, v in recalls_dict.items():
                full_recalls_dict[f"{val_set_name}_{k}"] = v

        self.log_dict(
            full_recalls_dict,
            prog_bar=True,
            logger=True,
        )
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for metric, value in full_recalls_dict.items():
            table.add_row([metric, f"{value:.4f}"])

        print(f"\nResults for {val_set_name}:")
        print(table)
        return full_recalls_dict

    def state_dict(self):
        sd = self.student.state_dict()
        return sd
