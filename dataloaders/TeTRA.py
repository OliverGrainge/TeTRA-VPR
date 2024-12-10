from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from torchvision import transforms as T
from transformers import get_cosine_schedule_with_warmup

from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.utils.TeTRA.losses import get_loss, get_miner
from models.transforms import get_transform
from matching.match_cosine import match_cosine
from matching.match_hamming import match_hamming
from dataloaders.utils.TeTRA.distances import HammingDistance


import argparse
from config import DataConfig, TeTRAConfig, ModelConfig


def symmetric_binarize(x):
    return torch.where(x > 0.0, torch.ones_like(x), -torch.ones_like(x))


class TeTRA(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size=32,
        image_size=[224, 224],
        num_workers=4,
        val_set_names=["pitts30k_val", "msls_val"],
        cities=["London", "Melbourne", "Boston"],
        lr=0.0001,
        img_per_place=4,
        min_img_per_place=4,
    ):
        super().__init__()
        # Model parameters
        self.lr = lr
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.cities = cities
        self.batch_acc = []
        # full precision loss and miner
        self.fp_loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
        )
        self.fp_miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

        # quantization aware loss and miner
        self.q_loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=HammingDistance()
        )
        self.q_miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=HammingDistance())

        self.model = model

        # Data parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []

        # Train and valid transforms
        self.train_transform = get_transform(
            augment_level="Light", image_size=image_size
        )
        self.valid_transform = get_transform(
            augment_level="None", image_size=image_size
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
            self.reload()
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import \
                        PittsburghDataset

                    self.val_datasets.append(
                        PittsburghDataset(
                            which_ds=val_set_name, input_transform=self.valid_transform
                        )
                    )
                elif val_set_name.lower() == "msls_val":
                    from dataloaders.val.MapillaryDataset import MSLS

                    self.val_datasets.append(MSLS(input_transform=self.valid_transform))
                elif val_set_name.lower() == "nordland":
                    from dataloaders.val.NordlandDataset import NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.valid_transform)
                    )
                elif val_set_name.lower() == "sped":
                    from dataloaders.val.SPEDDataset import SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=self.valid_transform)
                    )
                elif (
                    "sf_xl" in val_set_name.lower()
                    and "val" in val_set_name.lower()
                    and "small" in val_set_name.lower()
                ):
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(
                            which_ds="sf_xl_small_val", input_transform=self.transform
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
                            which_ds="sf_xl_small_test", input_transform=self.transform
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        warmup_steps = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _fp_loss_func(self, descriptors, labels): 
        miner_outputs = self.fp_miner(descriptors, labels)
        loss = self.fp_loss_fn(descriptors, labels, miner_outputs)

        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

        self.batch_acc.append(batch_acc)
        self.log('fp_b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    

    def _q_loss_func(self, descriptors, labels):
        q_descriptors = symmetric_binarize(descriptors)
        miner_outputs = self.q_miner(q_descriptors, labels)
        loss = self.q_loss_fn(descriptors, labels, miner_outputs)

        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

        self.batch_acc.append(batch_acc)
        self.log('q_b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss


    def training_step(self, batch, batch_idx): 
        places, labels = batch 
        BS, N, ch, h, w = places.shape 

        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)
        
        fp_loss = self._fp_loss_func(descriptors, labels)
        q_loss = self.q_loss_func(descriptors, labels)

        loss = fp_loss + q_loss
        self.log('fp_loss', fp_loss, prog_bar=True, logger=True)
        self.log('q_loss', q_loss, prog_bar=True, logger=True)
        self.log('loss', loss, prog_bar=True, logger=True)
        return loss


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

                fp_recalls_dict, _, search_time = match_cosine(
                    **set_outputs, 
                    num_references=val_dataset.num_references,
                    ground_truth=val_dataset.ground_truth,
                    k_values=[1, 5, 10]
                )

                full_recalls_dict[f"{val_set_name}_fp32_R1"] = fp_recalls_dict["R1"]
                full_recalls_dict[f"{val_set_name}_fp32_R5"] = fp_recalls_dict["R5"]
                full_recalls_dict[f"{val_set_name}_fp32_R10"] = fp_recalls_dict["R10"]
                full_recalls_dict[f"{val_set_name}_fp32_search_time"] = search_time

                q_recalls_dict, _, search_time = match_hamming(
                    **set_outputs, 
                    num_references=val_dataset.num_references,
                    ground_truth=val_dataset.ground_truth,
                    k_values=[1, 5, 10]
                )

                full_recalls_dict[f"{val_set_name}_q_R1"] = q_recalls_dict["R1"]
                full_recalls_dict[f"{val_set_name}_q_R5"] = q_recalls_dict["R5"]
                full_recalls_dict[f"{val_set_name}_q_R10"] = q_recalls_dict["R10"]
                full_recalls_dict[f"{val_set_name}_q_search_time"] = search_time

        self.log_dict(
            full_recalls_dict,
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
        return self.model.state_dict()


if __name__ == "__main__": 
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, TeTRAConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()


