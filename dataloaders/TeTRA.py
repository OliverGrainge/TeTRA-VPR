import argparse
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from transformers import get_cosine_schedule_with_warmup
import math 
import torch.optim as optim
import torch.nn.functional as F

import wandb
from config import ModelConfig, TeTRAConfig
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.utils.distances import HammingDistance, binarize
from dataloaders.utils.losses import get_loss, get_miner
from dataloaders.utils.schedulers import QuantScheduler
from matching.match_cosine import match_cosine
from matching.match_hamming import match_hamming
from models.transforms import get_transform
from dataloaders.eval.accuracy import get_val_dataset, DATASET_MAPPING
from dataloaders.eval.accuracy import get_recall_at_k, get_val_dataset

def _linear_schedule(step, total_steps):
    return min(step / total_steps, 1.0)

def _cosine_schedule(step, total_steps):
    s = min(step / total_steps, 1.0)
    return 0.5 * (1 - math.cos(math.pi * s))

def _logistic_schedule(step, total_steps, k=16, shift=6):
    # your existing “sigmoid” trick
    x = (step / total_steps) * k - shift
    return 1 / (1 + math.exp(-x))

def _no_schedule(step, total_steps):
    return 1.0

QUANT_SCHEDULES = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
    "logistic": _logistic_schedule,
    "none": _no_schedule,
}


class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()  # Returns -1 or 1
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Create gradient mask: 1 for inputs in [-1,1], 0 otherwise
        grad_mask = (input.abs() <= 1).float()
        return grad_output * grad_mask

def binarize(x):
    return BinarizeSTE.apply(x)

    
class TeTRA(pl.LightningModule):
    def __init__(
        self,
        model,
        train_dataset_dir,
        val_dataset_dir,
        batch_size=32,
        image_size=[322, 322],
        num_workers=4,
        val_set_names=["pitts30k"],
        cities=["London", "Melbourne", "Boston"],
        lr=0.0001,
        img_per_place=4,
        min_img_per_place=4,
        quant_schedule="logistic",
    ):
        super().__init__()
        # Model parameters
        self.lr = lr
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.cities = cities
        self.batch_acc = []
        # full precision loss and miner
        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
        )
        self.miner = miners.MultiSimilarityMiner(
            epsilon=0.1, distance=CosineSimilarity()
        )

        self.model = model

        # Data parameters
        self.base_path = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.quant_scheduler = QUANT_SCHEDULES[quant_schedule]

        # Train and valid transforms
        self.train_transform = get_transform(
            augmentation_level="Light", image_size=image_size
        )
        self.valid_transform = get_transform(
            augmentation_level="None", image_size=image_size
        )

        # Dataloader configs
        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": True,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": False,
        }

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            val_transform = get_transform(
                augmentation_level="None", image_size=self.image_size
            )
            for val_set_name in self.val_set_names:
                self.val_datasets.append(get_val_dataset(val_set_name, self.val_dataset_dir, val_transform, which_set="val"))
                
            for val_set_name in self.val_set_names:
                wandb.define_metric(f"{val_set_name}_binary_R1", summary="max")
                wandb.define_metric(f"{val_set_name}_float32_R1", summary="max")



    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            base_path=self.base_path,
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self): 
        backbone_params = self.model.backbone.parameters()
        aggregation_params = self.model.aggregation.parameters()

        param_groups = [
            {'params': backbone_params, 'lr': self.lr},  # Lower lr for backbone
            {'params': aggregation_params, 'lr': self.lr}  # Higher lr for aggregator
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.001)
    
        def lr_lambda(epoch):
            if (epoch + 1) < 3:
                return (epoch + 1) / 3  # Linear warmup
            else:
                return 0.3 ** sum(epoch >= milestone for milestone in [10, 20, 30])
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  # Step once per epoch
            }
        }
    

    def _fp_loss_func(self, descriptors, labels):
        miner_outputs = self.miner(descriptors, labels)
        loss = self.loss_fn(descriptors, labels, miner_outputs)

        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined / nb_samples)

        self.batch_acc.append(batch_acc)
        self.log(
            "fp_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def _q_loss_func(self, descriptors, labels):
        miner_outputs = self.miner(descriptors, labels)
        bin_descriptors = binarize(descriptors) 
        bin_descriptors = F.normalize(bin_descriptors, dim=-1)
        loss = self.loss_fn(bin_descriptors, labels, miner_outputs)

        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined / nb_samples)

        self.batch_acc.append(batch_acc)
        self.log(
            "binary_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss
    
    def _progressive_quant_scheduler(self):
        return self.quant_scheduler(self.global_step, self.trainer.estimated_stepping_batches)

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape

        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Split the batch in half
        split_size = images.shape[0] // 2
        images1, images2 = images[:split_size], images[split_size : 2 * split_size]
        # Process each half separately and concatenate
        descriptors1 = self(images1).to(torch.bfloat16)
        descriptors2 = self(images2).to(torch.bfloat16)
        descriptors = torch.cat([descriptors1, descriptors2], dim=0)

        fp_loss = self._fp_loss_func(descriptors, labels)
        q_loss = self._q_loss_func(descriptors, labels)
        qfactor = self._progressive_quant_scheduler()
        loss = (1 - qfactor) * fp_loss + qfactor * q_loss

        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(f'lr_group_{i}', param_group['lr'], logger=True)

        self.log("qfactor", qfactor, logger=True)
        self.log("fp_loss", fp_loss, logger=True)
        self.log("q_loss", q_loss, logger=True)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = {}
        for name in self.val_set_names:
            self.validation_outputs[name] = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        self.validation_outputs[self.val_set_names[dataloader_idx]].append(
            descriptors.detach().cpu()
        )
        return descriptors.detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""
        table = PrettyTable()
        table.field_names = ["Dataset", "Float32 R@1", "Binary R@1"]
        table.float_format = '.2'
        
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            descriptors = torch.concat(set_outputs, dim=0)

            recall_float32 = get_recall_at_k(descriptors, val_dataset, k_values=[1], precision="float32")[0]
            recall_binary = get_recall_at_k(descriptors, val_dataset, k_values=[1], precision="binary")[0]

            dataset_name = repr(val_dataset).replace("_val", "").replace("_test", "")
            self.log(f"{dataset_name}_binary_R1", recall_binary, prog_bar=True, logger=True)
            self.log(f"{dataset_name}_float32_R1", recall_float32, prog_bar=True, logger=True)
            
            table.add_row([dataset_name, f"{recall_float32:.1f}%", f"{recall_binary:.1f}%"])

        print("\nValidation Results:")
        print(table)

    def state_dict(self):
        # Override the state_dict method to return only the student model's state dict
        return self.model.state_dict()
