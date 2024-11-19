import glob
import json
import os
import random
import sys
import tarfile
from collections import defaultdict
from io import BytesIO

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from PIL import Image
from prettytable import PrettyTable
from pytorch_metric_learning import miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import get_cosine_schedule_with_warmup

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from matching.global_cosine_sim import global_cosine_sim
from models.helper import get_model, get_preset_transform
from dataloaders.train.DistillDataset import DistillDataset, TarImageDataset, JPGDataset
from utils.transforms import get_val_transform, get_train_transform
from dataloaders.utils.Distill import get_attn, remove_hooks, get_feature_dim, L2Norm, freeze_model

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
        print("=" * 100)
        print(
            "Teacher and student have the same descriptor dimension. No adaptation needed."
        )
        print("=" * 100)
        return nn.Identity()
    else:
        print("=" * 100)
        print(
            f"Teacher and student have different descriptor dimensions. Adapting student descriptor dimension to match teacher descriptor dimension: {student_dim} -> {teacher_dim}."
        )
        print("=" * 100)
        fc = nn.Linear(student_dim, teacher_dim, bias=False)
        mlp = nn.Sequential(fc, L2Norm())
        return mlp



class QLambdaScheduler:
    def __init__(self, module, max_steps, range=(-6, 8)):
        self.module = module
        self.max_steps = max_steps
        self.lambda_value = torch.ones(1)
        self.step_count = torch.zeros(1)
        self.range = range

    def step(self):
        self.step_count += 1
        self.lambda_value = torch.sigmoid(
            (
                self.step_count / self.max_steps * (self.range[1] - self.range[0])
                + self.range[0]
            )
        )
        for module in self.module.modules():
            if hasattr(module, "q_lambda"):
                module.q_lambda = self.lambda_value

    def get_lambda(self):
        return self.lambda_value


class VPRDistill(pl.LightningModule):
    def __init__(
        self,
        data_directory,
        student_backbone_arch="ResNet50",
        student_agg_arch="MixVPR",
        teacher_preset="EigenPlaces",
        augment_type="LightAugment", 
        matching_function=global_cosine_sim,
        use_attention=False,
        weight_decay_scale=0.05,
        weight_decay_schedule="staged_linear",
        batch_size=32,
        num_workers=4,
        image_size=224,
        lr=1e-3,
        mse_loss_scale=1000,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.lr = lr
        self.data_directory = data_directory
        self.teacher_preset = teacher_preset
        self.val_set_names = val_set_names
        self.backbone_arch = student_backbone_arch
        self.matching_function = matching_function
        self.use_attention = use_attention
        self.weight_decay_scale = weight_decay_scale
        self.weight_decay_schedule = weight_decay_schedule
        self.mse_loss_scale = mse_loss_scale
        self.teacher = get_model(preset=teacher_preset)
        self.augment_type = augment_type
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            out_dim=get_feature_dim(self.teacher, get_preset_transform(self.teacher_preset)),
            image_size=image_size,
        )
        self.adapter = adapt_descriptors_dim(
            self.teacher,
            get_preset_transform(self.teacher_preset),
            self.student,
            get_train_transform(self.augment_type, self.image_size),
        )

        freeze_model(self.teacher)
        print(self.student)
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset

                    self.val_datasets.append(
                        PittsburghDataset(
                            which_ds=val_set_name,
                            input_transform=get_val_transform(self.image_size),
                        )
                    )
                elif val_set_name.lower() == "msls_val":
                    from dataloaders.val.MapillaryDataset import MSLS

                    self.val_datasets.append(
                        MSLS(input_transform=get_val_transform(self.image_size))
                    )
                elif val_set_name.lower() == "nordland":
                    from dataloaders.val.NordlandDataset import NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.transform)
                    )
                elif val_set_name.lower() == "sped":
                    from dataloaders.val.SPEDDataset import SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=get_val_transform(self.image_size))
                    )
                elif (
                    "sf_xl" in val_set_name.lower()
                    and "val" in val_set_name.lower()
                    and "small" in val_set_name.lower()
                ):
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(
                            which_ds="sf_xl_small_val",
                            input_transform=get_val_transform(self.image_size),
                        )
                    )
                elif (
                    "sf_xl" in val_set_name.lower()
                    and "test" in val_set_name.lower()
                    and "small" in val_set_name.lower()
                ):
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(
                            which_ds="sf_xl_small_test",
                            input_transform=get_val_transform(self.image_size),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )

    def forward(self, x):
        return self.student(x)

    def _compute_features(self, student_images, teacher_images, use_attn):
        B = student_images.shape[0]
        if use_attn: 
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

            return student_features, teacher_features, student_attn, teacher_attn
        else:
            teacher_features = self.teacher(teacher_images)
            student_features = self(student_images)
            student_features["global_desc"] = self.adapter(
                student_features["global_desc"]
            )
            student_attn = None 
            teacher_attn = None 

            return student_features, teacher_features, None, None
    

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


    def _compute_cosine_loss(student_features, teacher_features): 
        cosine_loss = (1 - F.cosine_similarity(teacher_features, student_features)) ** 2
        return cosine_loss.mean()

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
            if hasattr(self.student.backbone, "q_lambda"):
                
        


    def _setup_schedulers(self, optimizer): 
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        weight_decay_scheduler = self._setup_weight_decay_scheduler()
        quant_scheduler = self._setup_q_lambda_scheduler()
        self.schedulers = {"lr": lr_scheduler, "weight_decay": weight_decay_scheduler,"quant": quant_scheduler}
    
    def _step_schedulers(self): 
        for param_name, schd in self.schedulers.items(): 
            schd.step()
            self.log(param_name, schd.get_last_lr()[0])


    def training_step(self, batch, batch_idx):
        student_images, teacher_images = batch
        student_features, teacher_features, student_attn, teacher_attn = self._compute_features(student_images, teacher_images)
        euc_loss = self.mse_loss_scale * self._compute_euclidian_loss(teacher_features, student_features)
        cos_loss = self._compute_cosine_loss(teacher_features, student_features)

        attn_loss = torch.tensor(0.0, device=cos_loss.device) 
        if self.use_attention: 
            attn_loss = self.mse_loss_scale * self._compute_attn_loss(teacher_attn, student_attn)
        total_loss = euc_loss + cos_loss + attn_loss

        self._weight_decay(batch_idx)
        self._schedulers_step(batch_idx)

        self.log("Cosine Loss", cos_loss)
        self.log("Euclidian Loss", euc_loss)
        self.log("Attention Loss", attn_loss)
        self.log("Total Loss", total_loss, prog_bar=True)
        return total_loss
            


    def configure_optimizers(self):

        if isinstance(self.student.backbone, Qmodel):
            optimizer = optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=0.0
            )
        elif "vit" in self.backbone_arch.lower():
            optimizer = optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=0.05
            )
        else:
            optimizer = optim.AdamW(
                self.student.parameters(), lr=self.lr, weight_decay=0.0
            )

        return optimizer
    
    def _setup_quant_scheduler(self):
        return QLambdaScheduler(
            self.student.backbone, self.trainer.estimated_stepping_batches
        )

    def _setup_weight_decay_scheduler(self):
        total_steps = self.trainer.estimated_stepping_batches

        # Custom scheduler for reg_scale
        if self.weight_decay_schedule == "staged_linear":

            def decay_schedule(step, init_weight_decay_scale):
                start_step = 0.1 * total_steps
                end_step = 0.9 * total_steps
                if step < start_step:
                    return init_weight_decay_scale
                elif step > end_step:
                    return 0.0
                else:
                    return init_weight_decay_scale * (
                        1 - (step - start_step) / (end_step - start_step)
                    )

        elif self.weight_decay_schedule == "constant":

            def decay_schedule(step, init_weight_decay_scale):
                return init_weight_decay_scale

        elif self.weight_decay_schedule == "sigmoid":

            def decay_schedule(step, init_weight_decay_scale):
                return init_weight_decay_scale * torch.sigmoid(
                    torch.tensor(step / total_steps * 14 - 4)
                )

        elif self.weight_decay_schedule is None:

            def decay_schedule(step, init_weight_decay_scale):
                return 0.0

        else:
            raise NotImplementedError(
                f"Weight decay schedule {self.weight_decay_schedule} not implemented"
            )

        class WeightDecayScheduler:
            def __init__(self, init_weight_decay_scale, decay_schedule):
                self.init_weight_decay_scale = init_weight_decay_scale
                self.weight_decay_scale = init_weight_decay_scale
                self.decay_schedule = decay_schedule
                self.step_count = 0

            def step(self):
                self.step_count += 1
                self.weight_decay_scale = self.decay_schedule(
                    self.step_count, self.init_weight_decay_scale
                )

            def get_last_weight_decay_scale(self):
                return self.weight_decay_scale

        weight_decay_scheduler = WeightDecayScheduler(
            self.weight_decay_scale, decay_schedule
        )
        return weight_decay_scheduler


    def train_dataloader(self):
        paths = os.listdir(self.data_directory)
        if not os.path.isdir(self.data_directory):
            raise ValueError(f"Invalid data directory: {self.data_directory}")

        if "*tar" in paths[0]:
            train_dataset = TarImageDataset(self.data_directory)
        else:
            train_dataset = JPGDataset(self.data_directory)

        dataset = DistillDataset(
            train_dataset,
            get_train_transform(self.augment_type, self.image_size),
            get_preset_transform(self.teacher_preset),
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
        descriptors["global_desc"] = self.adapter(descriptors["global_desc"])
        # store the outputs
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(
                value.detach().cpu()
            )
        return descriptors["global_desc"].detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""

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
