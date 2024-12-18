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
from models.transforms import get_transform
from matching.match_cosine import match_cosine
from models.helper import get_model
from models.transforms import get_transform


sys.path.append(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/../NeuroCompress"
    )
)

from NeuroPress.models.base import Qmodel


def adapt_descriptors_dim(
    teacher_model, teacher_transform, student_model, student_transform
):
    teacher_dim = get_feature_dim(teacher_model, teacher_transform)
    student_dim = get_feature_dim(student_model, student_transform)
    if teacher_dim == student_dim:
        print("=" * 1000)
        print(
            "Teacher and student have the same descriptor dimension. No adaptation needed."
        )
        print("=" * 1000)
        return nn.Identity()
    else:
        print("=" * 1000)
        print(
            f"Teacher and student have different descriptor dimensions. Adapting student descriptor dimension to match teacher descriptor dimension: {student_dim} -> {teacher_dim}."
        )
        print("=" * 1000)
        fc = nn.Linear(student_dim, teacher_dim, bias=False)
        mlp = nn.Sequential(fc, L2Norm())
        return mlp


class Distill(pl.LightningModule):
    def __init__(
        self,
        train_dataset_dir, 
        val_dataset_dir,
        student_backbone_arch="ResNet50",
        student_agg_arch="MixVPR",
        teacher_preset="EigenPlaces",
        augmentation_level="light",
        use_attention=False,
        weight_decay_init=0.05,
        weight_decay_schedule="staged_linear",
        batch_size=32,
        num_workers=4,
        image_size=224,
        lr=1e-3,
        mse_loss_mult=1000,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.lr = lr
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.teacher_preset = teacher_preset
        self.val_set_names = val_set_names
        self.backbone_arch = student_backbone_arch
        self.matching_function = match_cosine
        self.use_attention = use_attention
        self.weight_decay_init = weight_decay_init
        self.weight_decay_schedule = weight_decay_schedule
        self.mse_loss_mult = mse_loss_mult
        self.teacher = get_model(preset=teacher_preset)
        self.augmentation_level = augmentation_level
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            out_dim=get_feature_dim(
                self.teacher, get_transform(preset=self.teacher_preset)
            ),
            image_size=image_size,
        )
        self.adapter = adapt_descriptors_dim(
            self.teacher,
            get_transform(preset=self.teacher_preset),
            self.student,
            get_transform(augmentation_level=self.augmentation_level, image_size=self.image_size),
        )

        freeze_model(self.teacher)
        print(self.student)
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            val_transform = get_transform(augmentation_level="None", image_size=self.image_size)
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset30k
                    self.val_datasets.append(
                        PittsburghDataset30k(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "pitts250k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset250k
                    self.val_datasets.append(
                        PittsburghDataset250k(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "msls" in val_set_name.lower():
                    from dataloaders.val.MapillaryDataset import MSLS
                    self.val_datasets.append(
                        MSLS(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "nordland" in val_set_name.lower():
                    from dataloaders.val.NordlandDataset import NordlandDataset
                    self.val_datasets.append(
                        NordlandDataset(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "sped" in val_set_name.lower():
                    from dataloaders.val.SPEDDataset import SPEDDataset
                    self.val_datasets.append(
                        SPEDDataset(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val"))
                elif "essex" in val_set_name.lower():
                    from dataloaders.val.EssexDataset import EssexDataset
                    self.val_datasets.append(
                        EssexDataset(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "sanfrancicscosmall" in val_set_name.lower():
                    from dataloaders.val.SanFrancisco import SanFranciscoSmall
                    self.val_datasets.append(
                        SanFranciscoSmall(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "tokyo" in val_set_name.lower():
                    from dataloaders.val.Tokyo247 import Tokyo247
                    self.val_datasets.append(
                        Tokyo247(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                elif "cross" in val_set_name.lower():
                    from dataloaders.val.CrossSeasonDataset import CrossSeasonDataset
                    self.val_datasets.append(
                        CrossSeasonDataset(val_dataset_dir=self.val_dataset_dir, input_transform=val_transform, which_set="val")
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )

    def forward(self, x):
        return self.student(x)

    def _setup_schedulers(self, optimizer):
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        weight_decay_scheduler = WeightDecayScheduler(
            self.weight_decay_init,
            total_steps,
            schedule_type=self.weight_decay_schedule,
        )
        quant_scheduler = QuantScheduler(total_steps)
        return {
            "lr": lr_scheduler,
            "weight_decay": weight_decay_scheduler,
            "quant": quant_scheduler,
        }

    def _step_schedulers(self, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            for param_name, schd in self.schedulers.items():
                schd.step()
                self.log(param_name, schd.get_last_lr()[0], on_step=True)

    def configure_optimizers(self):
        if isinstance(self.student.backbone, Qmodel):
            optimizer = optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=0.0
            )  # The custom schedulers handle weight decay in this case
        else:
            optimizer = optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=0.05
            )

        self.schedulers = self._setup_schedulers(optimizer)
        return optimizer

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
                student_features["global_desc"],
                teacher_features["global_desc"],
                student_attn,
                teacher_attn,
            )
        else:
            teacher_features = self.teacher(teacher_images)
            student_features = self(student_images)
            student_features["global_desc"] = self.adapter(
                student_features["global_desc"]
            )
            student_attn = None
            teacher_attn = None

            return (
                student_features["global_desc"],
                teacher_features["global_desc"],
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

    def _weight_decay(self, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            if hasattr(self.student.backbone, "decay_weight"):
                decay_scale = self.schedulers["weight_decay"].get_last_lr()[0]
                lr = self.schedulers["lr"].get_last_lr()[0]
                self.student.backbone.decay_weight(
                    lr=lr,
                    weight_decay_scale=decay_scale,
                )

    def _progressive_quant(self, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            q_lambda_val = self.schedulers["quant"].get_last_lr()[0]
            for module in self.student.modules():
                if hasattr(module, "q_lambda"):
                    module.q_lambda = q_lambda_val

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

        self._weight_decay(batch_idx)
        self._progressive_quant(batch_idx)
        self._step_schedulers(batch_idx)

        self.log("Cosine Loss", cos_loss, on_step=True)
        self.log("Euclidian Loss", euc_loss, on_step=True)
        self.log("Attention Loss", attn_loss, on_step=True)
        self.log("Total Loss", total_loss, on_step=True, prog_bar=True)
        return total_loss

    def train_dataloader(self):
        paths = os.listdir(self.train_dataset_dir)
        if not os.path.isdir(self.train_dataset_dir):
            raise ValueError(f"Invalid data directory: {self.train_dataset_dir}")

        if "*tar" in paths[0]:
            train_dataset = TarImageDataset(self.train_dataset_dir)
        else:
            train_dataset = JPGDataset(self.train_dataset_dir)

        dataset = DistillDataset(
            dataset=train_dataset,
            student_transform=get_transform(augmentation_level=self.augmentation_level, image_size=self.image_size),
            teacher_transform=get_transform(preset=self.teacher_preset),
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        descriptors["global_desc"] = self.adapter(descriptors["global_desc"])
        # store the outputs
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(
                value.detach().cpu()
            )
        return descriptors["global_desc"].detach().cpu()

    def on_validation_epoch_end(self):
        full_recalls_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _, _ = self.matching_function(
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
        return self.student.state_dict()
