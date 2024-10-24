from collections import defaultdict

import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from matching.global_cosine_sim import global_cosine_sim

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class GSVCities(pl.LightningModule):
    def __init__(
        self,
        config,
        model,
        batch_size=32,
        shuffle_all=False,
        image_size=(480, 640),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val", "msls_val"],
        search_precision="float32",
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        matching_function=global_cosine_sim,
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
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = config["miner_margin"]
        self.faiss_gpu = config["faiss_gpu"]
        self.search_precision = search_precision
        self.img_per_place = config["img_per_place"]
        self.min_img_per_place = config["min_img_per_place"]
        self.cities = config["cities"]
        self.shuffle_all = config["shuffle_all"]
        self.matching_function = matching_function
        self.batch_acc = []
        self.loss_fn = utils.get_loss(self.loss_name)
        self.miner = utils.get_miner(self.miner_name, self.miner_margin)

        self.model = model

        # Data parameters
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
        return [optimizer], [scheduler]

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
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
        loss = self.loss_function(descriptors["global_desc"], labels)
        self.log("loss", loss)
        return {"loss": loss}

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

        results_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _ = self.matching_function(
                **set_outputs,
                num_references=val_dataset.num_references,
                ground_truth=val_dataset.ground_truth,
            )
            self.log_dict(
                recalls_dict,
                prog_bar=False,
                logger=True,
            )
            results_dict[val_set_name] = recalls_dict
        return results_dict
