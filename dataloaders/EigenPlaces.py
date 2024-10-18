import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataloaders.train.EigenPlacesDataset import EigenPlacesDataset
from dataloaders.train.Utils import augmentations

import utils

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


# Load config
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, loop_count=1000, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.loop_count = loop_count

    def __len__(self):
        return len(self.dataset) * self.loop_count

    def __iter__(self):
        while True:
            for _ in range(self.loop_count):
                for batch in super().__iter__():
                    yield batch


def cosine_sim(
    x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output


def get_output_dim(image_size, model):
    image = torch.randn(3, *image_size).to(next(model.parameters()).device)
    out = model(image[None, :])
    return out.shape[1]


class EigenPlaces(pl.LightningModule):
    def __init__(
        self,
        config,
        model,
        image_size=(512, 512),
        batch_size=32,
        val_set_names=["pitts30k_val"],
        mean_std=IMAGENET_MEAN_STD,
        num_workers=8,
        search_precision="float32",
    ):
        super().__init__()

        # Get backbone and aggregator
        self.model = model

        # Training hyperparameters
        self.lr = config["lr"]
        self.classifiers_lr = config["classifiers_lr"]

        # FAISS settings
        self.faiss_gpu = False
        self.search_precision = search_precision

        # Lateral and Frontal Loss Scaling
        self.lambda_lat = config["lambda_lat"]
        self.lambda_front = config["lambda_front"]

        # Data configuration
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_set_names = val_set_names
        self.output_dim = get_output_dim(self.image_size, model)
        self.M = config["M"]
        self.N = config["N"]
        self.s = config["s"]
        self.m = config["m"]
        self.focal_dist = config["focal_dist"]
        self.min_images_per_class = config["min_images_per_class"]
        self.visualize_classes = config["visualize_classes"]
        self.groups_num = self.N * self.N
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.num_workers = num_workers
        self.brightness = config["brightness"]
        self.contrast = config["contrast"]
        self.hue = config["hue"]
        self.saturation = config["saturation"]
        self.random_resized_crop = config["random_resized_crop"]
        self.dataset_size = config["dataset_size"]
        self.weight_decay = config["weight_decay"]

        # Train and valid transforms
        self.train_transform = T.Compose(
            [
                augmentations.DeviceAgnosticColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                ),
                augmentations.DeviceAgnosticRandomResizedCrop(
                    self.image_size, scale=[1 - self.random_resized_crop, 1]
                ),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.groups = [
            EigenPlacesDataset(
                dataset_size=self.dataset_size,
                M=self.M,
                N=self.N,
                focal_dist=self.focal_dist,
                current_group=n // 2,
                min_images_per_class=self.min_images_per_class,
                angle=[0, 90][n % 2],
                visualize_classes=self.visualize_classes,
            )
            for n in range(self.groups_num * 2)
        ]

        # Group-specific classifiers
        self.classifiers = [
            MarginCosineProduct(
                self.output_dim, group.num_classes(), s=self.s, m=self.m
            )
            for group in self.groups
        ]
        for cls in self.classifiers:
            cls = cls.to("cuda")
        print(f"number of classes {[g.num_classes() for g in self.groups]}")
        print(
            f"The {len(self.groups)} groups have respectively the following "
            f"number of images {[g.get_images_num() for g in self.groups]}"
        )
        self.criterion = torch.nn.CrossEntropyLoss()

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
            "drop_last": True,
            "pin_memory": True,
            "shuffle": False,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": False,
        }

        self.automatic_optimization = False

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                print(val_set_name)
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

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        # Separate linear layer parameters (for weight decay) and others (without weight decay)
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if isinstance(
                self.model.get_submodule(name.split(".")[0]), torch.nn.Linear
            ):
                decay_params.append(param)  # Apply weight decay to linear layers
            else:
                no_decay_params.append(param)  # No weight decay for other layers

        # Create optimizer groups
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Optimizer for the model with weight decay only on linear layers
        model_opt = torch.optim.Adam(param_groups, lr=self.lr)

        # Optimizers for the classifiers (without weight decay)
        classifiers_optimizers = [
            torch.optim.Adam(classifier.parameters(), lr=self.classifiers_lr)
            for classifier in self.classifiers
        ]

        # Combine optimizers
        opt = [model_opt] + classifiers_optimizers
        return opt

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        model_opt = opt[0]
        classifier_opts = opt[1:]

        model_opt.zero_grad()

        for dataset_key, b in batch.items():
            current_dataset_num, i = dataset_key
            images, targets, _ = b
            images = self.train_transform(images)
            descriptors = self(images)
            classifier_opts[current_dataset_num + i].zero_grad()
            output = self.classifiers[current_dataset_num + i](descriptors["global_desc"], targets)
            loss = self.criterion(output, targets)
            if i == 0:
                loss *= self.lambda_lat
                self.log("lateral loss", loss)
            else:
                loss *= self.lambda_front
                self.log("frontal loss", loss)
            self.manual_backward(loss)
            classifier_opts[current_dataset_num + i].step()

        model_opt.step()

    def train_dataloader(self):

        # Dataloaders for different datasets in different groups
        """
        loaders = {}
        for group_num in range(self.groups_num):
            loaders[group_num] = infinite_dataloader(DataLoader(self.groups[group_num], **self.train_loader_config))
        """

        current_dataset_num = (self.current_epoch % self.groups_num) * 2
        loaders = {}
        for i in range(2):
            loaders[(current_dataset_num, i)] = InfiniteDataLoader(
                self.groups[current_dataset_num + i], **self.train_loader_config
            )

        return loaders

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.validation_outputs.append(descriptors["global_desc"].detach().cpu())
        return None

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""
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

            recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision=self.search_precision,
            )

            self.log(f"{val_set_name}/R1", recalls_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls_dict[5], prog_bar=False, logger=True)
            self.log(
                f"{val_set_name}/R10", recalls_dict[10], prog_bar=False, logger=True
            )

            del r_list, q_list, feats, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs.clear()
        print("\n\n")
