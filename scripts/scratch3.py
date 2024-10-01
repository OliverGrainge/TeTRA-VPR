import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

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
        # val_set_names=["pitts30k_val", "msls_val"],
        val_set_names=["pitts30k_val"],
        search_precision="float32",
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
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

        self.batch_acc = []
        margin = 0.1
        self.coarse_loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
        )
        self.coarse_miner = miners.MultiSimilarityMiner(
            epsilon=0.2, distance=CosineSimilarity()
        )
        self.fine_loss_fn = losses.MultiSimilarityLoss(
            alpha=1.0, beta=100, base=0.0, distance=CosineSimilarity()
        )
        self.fine_miner = miners.MultiSimilarityMiner(
            epsilon=-0.05, distance=CosineSimilarity()
        )
        # self.fine_miner = miners.TripletMarginMiner(margin=0.05, type_of_triplets="hard")
        # self.fine_loss_fn = losses.TripletMarginLoss(margin=0.05,
        #                swap=False,
        #                smooth_loss=False,
        #                triplets_per_anchor="all")

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

    def coarse_loss_function(self, descriptors, labels):
        if self.coarse_miner is not None:
            miner_outputs = self.coarse_miner(descriptors, labels)
            # print("coarse negative pairs: ", miner_outputs[2].shape[0], "Coarse positive pairs: ", miner_outputs[0].shape[0])
            loss = self.coarse_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.coarse_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc_coarse",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def fine_loss_function(self, descriptors, labels):
        if self.fine_miner is not None:
            miner_outputs = self.fine_miner(descriptors, labels)
            # print("fine negative pairs: ", miner_outputs[2].shape[0], "fine positive pairs: ", miner_outputs[0].shape[0])
            loss = self.fine_loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.fine_loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc_coarse",
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
        desc_coarse, desc_fine = self(images)
        coarse_loss = self.coarse_loss_function(desc_coarse, labels)
        fine_loss = self.fine_loss_function(desc_fine, labels)
        loss = coarse_loss / 2 + fine_loss / 2
        self.log("coarse loss", coarse_loss)
        self.log("fine loss", fine_loss)
        self.log("loss", loss)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.coarse_validation_outputs = []
        self.fine_validation_outputs = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        coarse_desc, fine_desc = self(places)
        # store the outputs
        self.coarse_validation_outputs.append(coarse_desc.detach().cpu())
        self.fine_validation_outputs.append(fine_desc.detach().cpu())
        return coarse_desc.detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(self.val_datasets) == 1:
            coarse_val_step_outputs = [self.coarse_validation_outputs]
            fine_val_step_outputs = [self.fine_validation_outputs]
        else:
            coarse_val_step_outputs = self.coarse_validation_outputs
            fine_val_step_outputs = self.fine_validation_outputs

        for i, (val_set_name, val_dataset) in enumerate(
            zip(self.val_set_names, self.val_datasets)
        ):
            coarse_feats = torch.concat(coarse_val_step_outputs[i], dim=0)
            fine_feats = torch.concat(fine_val_step_outputs[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries
            r_list_coarse = coarse_feats[:num_references]
            q_list_coarse = coarse_feats[num_references:]
            r_list_fine = fine_feats[:num_references]
            q_list_fine = fine_feats[num_references:]

            recalls_dict_float, predictions = utils.get_validation_recalls(
                r_list=r_list_coarse,
                q_list=q_list_coarse,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="float32",
                desc="coarse stage",
            )

            self.log(
                f"{val_set_name}/coarse_R1",
                recalls_dict_float[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/coarse_R5",
                recalls_dict_float[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/coarse_R10",
                recalls_dict_float[10],
                prog_bar=False,
                logger=True,
            )

            recalls_dict_float, predictions = utils.get_validation_recalls(
                r_list=r_list_fine,
                q_list=q_list_fine,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="float32",
                desc="fine stage",
            )

            self.log(
                f"{val_set_name}/fine_R1",
                recalls_dict_float[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/fine_R5",
                recalls_dict_float[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/fine_R10",
                recalls_dict_float[10],
                prog_bar=False,
                logger=True,
            )

            recalls_dict_float, predictions = utils.get_validation_recalls_two_stage(
                r_list_binary=r_list_coarse,
                q_list_binary=q_list_coarse,
                r_list_float=r_list_fine,
                q_list_float=q_list_fine,
                k_values=[1, 5, 10, 15, 20, 25],
                k=20,
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
            )

            self.log(
                f"{val_set_name}/two_stage_R1",
                recalls_dict_float[1],
                prog_bar=False,
                logger=True,
            )

            del (
                r_list_coarse,
                r_list_fine,
                q_list_coarse,
                q_list_fine,
                coarse_feats,
                fine_feats,
                num_references,
                ground_truth,
            )

        # Clear the outputs after processing
        self.fine_validation_outputs.clear()
        self.coarse_validation_outputs.clear()
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


if __name__ == "__main__":
    import yaml

    from models.helper import get_model
    from parsers import get_args_parser

    parser = get_args_parser()
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = get_model(
        (224, 224),
        "resnet50",
        "mixvpr_two_step",
        config["Model"],
        normalize_output=True,
    )

    model_module = GSVCities(
        config["Training"]["GSVCities"],
        model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        mean_std={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        # val_set_names=args.val_set_names,
        val_set_names=["pitts30k_val"],
        search_precision=args.search_precision,
        loss_name=args.loss_name,
        miner_name=args.miner_name,
    )

    trainer = pl.Trainer(
        enable_progress_bar=True,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=100,
        # limit_train_batches=50
    )

    trainer.fit(model_module)
