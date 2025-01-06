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

from dataloaders.train.DistillDataset import (DistillDataset, JPGDataset,
                                              TarImageDataset)
from dataloaders.utils.Distill.attention import get_attn, remove_hooks
from dataloaders.utils.Distill.funcs import (L2Norm, freeze_model,
                                             get_feature_dim)
from dataloaders.utils.Distill.schedulers import (QuantScheduler,
                                                  WeightDecayScheduler)
from matching.match_cosine import match_cosine
from models.helper import get_model
from models.transforms import get_transform



class Distill(pl.LightningModule):
    def __init__(
        self,
        student_model, 
        teacher_model,
        train_dataset_dir,
        val_dataset_dir,
        augmentation_level="Severe",
        use_attention=True,
        weight_decay=0.01,
        batch_size=32,
        num_workers=4,
        image_size=224,
        lr=1e-3,
        mse_loss_mult=1000,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
         # Model-related attributes
        self.teacher = teacher_model
        self.student = student_model

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

        freeze_model(self.teacher)
        print(self.student)
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            val_transform = get_transform(
                augmentation_level="None", image_size=self.image_size
            )
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import \
                        PittsburghDataset30k

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
            num_warmup_steps=int(0.05 * num_training_steps),  # You can adjust this based on your needs
            num_training_steps=num_training_steps
        )
        return [optimizer], [scheduler]

    def _compute_features(self, student_images, teacher_images, use_attention):
        B = student_images.shape[0]
        if use_attention:
            teacher_attn, teacher_hooks = get_attn(self.teacher)
            student_attn, student_hooks = get_attn(self.student)
            teacher_features = self.teacher(teacher_images)
            student_features = self(student_images)
            teacher_attn = torch.vstack(teacher_attn)
            student_attn = torch.vstack(student_attn)

            # B * D, H, N, N
            teacher_attn = teacher_attn.view(
                B,
                -1,
                teacher_attn.shape[-3],
                teacher_attn.shape[-2],
                teacher_attn.shape[-1],
            )
            student_attn = student_attn.view(
                B,
                -1,
                student_attn.shape[-3],
                student_attn.shape[-2],
                student_attn.shape[-1],
            )

            remove_hooks(teacher_hooks)
            remove_hooks(student_hooks)
            return (
                student_features,
                teacher_features,
                student_attn,
                teacher_attn,
            )
        else:
            teacher_features = self.teacher(teacher_images)
            student_features = self(student_images)
            student_attn = None
            teacher_attn = None

            return (
                student_features,
                teacher_features,
                None,
                None,
            )

    @staticmethod
    def _compute_attn_loss(student_attn, teacher_attn):
        # B, D, H, N, N
        if teacher_attn.shape[1] != student_attn.shape[1]:
            # have different depths
            assert (
                teacher_attn.shape[1] > student_attn.shape[1]
            ), "teacher attention must be deeper or equal to the depth of the than student attention"
            teacher_attn_idxs = (
                torch.arange(student_attn.shape[1]) * teacher_attn.shape[1]
            ) / student_attn.shape[1].floor().long()
            teacher_attn = teacher_attn[teacher_attn_idxs, :, :, :]

        if teacher_attn.shape[2] != student_attn.shape[2]:
            # have different number of attention heads
            teacher_attn = teacher_attn.mean(2)
            student_attn = student_attn.mean(2)

        if teacher_attn.shape[-1] != student_attn.shape[-1]:
            # have different number of tokens
            B, D, H = student_attn.shape[:3]
            if len(student_attn.shape) == 5:
                student_attn = F.interpolate(
                    student_attn.view(
                        B * D * H, 1, student_attn.shape[-2], student_attn.shape[-1]
                    ),
                    size=teacher_attn.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(B, D, H, teacher_attn.shape[-2], teacher_attn.shape[-1])
            else:
                student_attn = F.interpolate(
                    student_attn,
                    size=teacher_attn.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        return F.mse_loss(teacher_attn, student_attn)

    @staticmethod
    def _compute_cosine_loss(student_features, teacher_features):
        cosine_loss = (1 - F.cosine_similarity(teacher_features, student_features)) ** 2
        return cosine_loss.mean()

    @staticmethod
    def _compute_euclidian_loss(student_features, teacher_features):
        return F.mse_loss(teacher_features, student_features)

    def training_step(self, batch, batch_idx):
        student_images, teacher_images = batch
        student_features, teacher_features, student_attn, teacher_attn = (
            self._compute_features(student_images, teacher_images, self.use_attention)
        )
        euc_loss = self.mse_loss_mult * self._compute_euclidian_loss(
            teacher_features, student_features
        )
        cos_loss = self._compute_cosine_loss(teacher_features, student_features)

        attn_loss = torch.tensor(0.0, device=cos_loss.device)
        if self.use_attention:
            attn_loss = self.mse_loss_mult * self._compute_attn_loss(
                teacher_attn, student_attn
            )
        total_loss = euc_loss + cos_loss + attn_loss

        self.log("Cosine Loss", cos_loss, on_step=True)
        self.log("Euclidian Loss", euc_loss, on_step=True)
        self.log("Attention Loss", attn_loss, on_step=True)
        self.log("Total Loss", total_loss, on_step=True, prog_bar=True)
        return total_loss

    def train_dataloader(self):
        paths = os.listdir(self.train_dataset_dir)
        if not os.path.isdir(self.train_dataset_dir):
            raise ValueError(f"Invalid data directory: {self.train_dataset_dir}")

        train_dataset = JPGDataset(self.train_dataset_dir)

        dataset = DistillDataset(
            dataset=train_dataset,
            student_transform=get_transform(
                augmentation_level="Severe", image_size=self.image_size
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
            self.validation_outputs[name] = defaultdict(list)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        descriptors = self(places)
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(
                value.detach().cpu()
            )
        return descriptors.detach().cpu()

    def on_validation_epoch_end(self):
        full_recalls_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _, _ = match_cosine(
                **set_outputs,
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
        # Override the state_dict method to return only the student model's state dict
        self.student.train() # remove the qweight buffers
        return self.student.state_dict()
