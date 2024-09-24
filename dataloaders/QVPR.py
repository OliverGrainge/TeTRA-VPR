import math

import numpy as np
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (CosineSimilarity,
                                               DotProductSimilarity)

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class QVPR(pl.LightningModule):
    def __init__(
        self,
        config,
        model,
        max_epochs,
        batch_size=32,
        shuffle_all=False,
        image_size=(480, 640),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val", "msls_val"],
        search_precision="float32",
    ):
        super().__init__()
        # Model parameters
        self.lr = config["lr"]
        self.optimizer_type = config["optimizer"]
        self.weight_decay = config["weight_decay"]
        self.momentum = config["momentum"]
        self.warmup_steps = config["warmup_steps"]
        self.milestones = config["milestones"]
        self.lr_mult = config["lr_mult"]
        self.miner_margin = config["miner_margin"]
        self.faiss_gpu = config["faiss_gpu"]
        self.search_precision = search_precision
        self.img_per_place = config["img_per_place"]
        self.min_img_per_place = config["min_img_per_place"]
        self.cities = config["cities"]
        self.shuffle_all = config["shuffle_all"]

        self.batch_acc = []
        self.float_loss_fn = utils.get_loss("MultiSimilarityLoss")
        self.float_miner = utils.get_miner("MultiSimilarityMiner", self.miner_margin)
        self.binary_loss_fn = utils.get_loss("HammingSTEMultiSimilarityLoss")
        self.binary_miner = utils.get_miner(
            "HammingSTEMultiSimilarityMiner", self.miner_margin
        )

        self.model = model

        # Data parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.show_data_stats = show_data_stats

        # Train and valid transforms
        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        # Dataloader configs
        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": self.shuffle_all,
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
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )
            if self.show_data_stats:
                self.print_stats()

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
        if self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_type} not recognized.")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )

        self.gamma = 0.0
        self.total_steps = len(self.train_dataloader()) * self.max_epochs
        return [optimizer], [scheduler]

    def on_train_start(self):
        # Optionally, log total_steps for verification
        if self.trainer.is_global_zero:
            print(f"Total steps per process: {self.total_steps}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update gamma at the end of each training batch
        self.update_gamma()

    def update_gamma(self):
        current_step = self.trainer.global_step
        if self.total_steps == 0:
            step_ratio = 0.0
        else:
            step_ratio = current_step / self.total_steps
        angle = step_ratio * math.pi - (math.pi / 2)
        self.gamma = (math.sin(angle) + 1) / 2
        self.log("gamma", self.gamma, prog_bar=True, logger=True)

    def float_loss_function(self, descriptors, labels):
        if self.float_miner is not None:
            miner_outputs = self.float_miner(descriptors, labels)
            loss = self.float_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.float_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def binary_loss_function(self, descriptors, labels):
        if self.binary_miner is not None:
            miner_outputs = self.binary_miner(descriptors, labels)
            loss = self.binary_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.binary_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)
        float_loss = self.float_loss_function(descriptors, labels)
        binary_loss = self.binary_loss_function(descriptors, labels)
        loss = (1 - self.gamma) * float_loss + self.gamma * binary_loss
        self.log("float_loss", float_loss)
        self.log("binary_loss", binary_loss)
        self.log("loss", loss)
        self.log("gamma", self.gamma)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        self.validation_outputs.append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(self.val_datasets) == 1:
            val_step_outputs = [self.validation_outputs]
        else:
            val_step_outputs = self.validation_outputs

        for i, (val_set_name, val_dataset) in enumerate(
            zip(self.val_set_names, self.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries
            r_list = feats[:num_references]
            q_list = feats[num_references:]

            float_recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="float32",
            )

            binary_recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="binary",
            )

            self.log(
                f"{val_set_name}/float_R1",
                float_recalls_dict[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R5",
                float_recalls_dict[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R10",
                float_recalls_dict[10],
                prog_bar=False,
                logger=True,
            )

            self.log(
                f"{val_set_name}/binary_R1",
                binary_recalls_dict[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R5",
                binary_recalls_dict[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R10",
                binary_recalls_dict[10],
                prog_bar=False,
                logger=True,
            )

            del r_list, q_list, feats, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs.clear()
        print("\n\n")

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.add_row(["# of cities", f"{len(self.cities)}"])
        table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
        table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))

        table = PrettyTable()
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))

        table = PrettyTable()
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
























#===============================================================================================

class QVPR2(pl.LightningModule):
    def __init__(
        self,
        config,
        model,
        max_epochs,
        batch_size=32,
        shuffle_all=False,
        image_size=(480, 640),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val", "msls_val"],
        search_precision="float32",
    ):
        super().__init__()
        # Model parameters
        self.lr = config["lr"]
        self.optimizer_type = config["optimizer"]
        self.weight_decay = config["weight_decay"]
        self.momentum = config["momentum"]
        self.warmup_steps = config["warmup_steps"]
        self.milestones = config["milestones"]
        self.lr_mult = config["lr_mult"]
        self.miner_margin = config["miner_margin"]
        self.faiss_gpu = config["faiss_gpu"]
        self.search_precision = search_precision
        self.img_per_place = config["img_per_place"]
        self.min_img_per_place = config["min_img_per_place"]
        self.cities = config["cities"]
        self.shuffle_all = config["shuffle_all"]

        self.batch_acc = []
        self.float_loss_fn = utils.get_loss("MultiSimilarityLoss")
        self.float_miner = utils.get_miner("MultiSimilarityMiner", margin=self.miner_margin)
        self.binary_loss_fn = utils.get_loss("HammingSTEMultiSimilarityLoss")
        self.binary_miner = utils.get_miner(
            "HammingSTEMultiSimilarityMiner", margin=self.miner_margin
        )

        self.finegrain_loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
        )
        self.finegrain_miner = miners.MultiSimilarityMiner(epsilon=0.3, distance=CosineSimilarity())
        self.model = model

        # Data parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.show_data_stats = show_data_stats

        # Train and valid transforms
        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        # Dataloader configs
        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": self.shuffle_all,
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
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )
            if self.show_data_stats:
                self.print_stats()

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
        if self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_type} not recognized.")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )

        self.gamma = 0.0
        self.total_steps = len(self.train_dataloader()) * self.max_epochs
        return [optimizer], [scheduler]

    def on_train_start(self):
        # Optionally, log total_steps for verification
        if self.trainer.is_global_zero:
            print(f"Total steps per process: {self.total_steps}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update gamma at the end of each training batch
        self.update_gamma()

    def update_gamma(self):
        current_step = self.trainer.global_step
        if self.total_steps == 0:
            step_ratio = 0.0
        else:
            step_ratio = current_step / self.total_steps
        angle = step_ratio * math.pi - (math.pi / 2)
        self.gamma = (math.sin(angle) + 1) / 2
        self.log("gamma", self.gamma, prog_bar=True, logger=True)

    def float_loss_function(self, descriptors, labels):
        if self.float_miner is not None:
            miner_outputs = self.float_miner(descriptors, labels)
            loss = self.float_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.float_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def binary_loss_function(self, descriptors, labels):
        if self.binary_miner is not None:
            miner_outputs = self.binary_miner(descriptors, labels)
            loss = self.binary_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.binary_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss
    
    def finetune_loss_function(self, descriptors, labels):
        miner_outputs = self.finegrain_miner(descriptors, labels)
        loss = self.finegrain_loss_fn(descriptors, labels, miner_outputs)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)
        fine_descriptors = None
        if type(descriptors) == tuple: 
            descriptors, fine_descriptors = descriptors

        float_loss = self.float_loss_function(descriptors, labels)
        binary_loss = self.binary_loss_function(descriptors, labels)
        loss = (1 - self.gamma) * float_loss + self.gamma * binary_loss

        if fine_descriptors is not None: 
            fine_loss = self.finetune_loss_function(fine_descriptors, labels)
            self.log("finegrain_loss", fine_loss)
        loss += fine_loss  
        self.log("float_loss", float_loss)
        self.log("binary_loss", binary_loss)
        self.log("loss", loss)
        self.log("gamma", self.gamma)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs_float = []
        self.validation_outputs_binary = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        self.validation_outputs_binary.append(descriptors[0].detach().cpu())
        self.validation_outputs_float.append(descriptors[1].detach().cpu())
        return descriptors[0].detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(self.val_datasets) == 1:
            val_step_outputs_float = [self.validation_outputs_float]
            val_step_outputs_binary = [self.validation_outputs_binary]
        else:
            val_step_outputs_float = self.validation_outputs_float
            val_step_outputs_binary = self.validation_outputs_binary

        for i, (val_set_name, val_dataset) in enumerate(
            zip(self.val_set_names, self.val_datasets)
        ):
            feats_bin = torch.concat(val_step_outputs_binary[i], dim=0)
            feats_float = torch.concat(val_step_outputs_binary[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries
            r_list_float = feats_float[:num_references]
            q_list_float = feats_float[num_references:]
            r_list_bin = feats_bin[:num_references]
            q_list_bin = feats_bin[num_references:]

            float_recalls_dict, predictions = utils.get_validation_recalls_two_stage(r_list_float, r_list_bin, q_list_float, q_list_bin, 10, gt=ground_truth)


            self.log(
                f"{val_set_name}/binary_R1",
                float_recalls_dict[1],
                prog_bar=False,
                logger=True,
            )

            del r_list_float, r_list_bin, q_list_float, q_list_bin, feats_bin, feats_float, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs_float.clear()
        self.validation_outputs_binary.clear()
        print("\n\n")

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.add_row(["# of cities", f"{len(self.cities)}"])
        table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
        table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))

        table = PrettyTable()
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))

        table = PrettyTable()
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))