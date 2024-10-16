import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class VPREval(pl.LightningModule):
    def __init__(
        self,
        model,
        transform,
        val_set_names=["pitts30k_val"],
        search_precision="float32",
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.model = model
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.search_precision = search_precision

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
                            which_ds=val_set_name, input_transform=self.transform
                        )
                    )
                elif val_set_name.lower() == "msls_val":
                    from dataloaders.val.MapillaryDataset import MSLS

                    self.val_datasets.append(MSLS(input_transform=self.transform))
                elif val_set_name.lower() == "nordland":
                    from dataloaders.val.NordlandDataset import NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.transform)
                    )
                elif val_set_name.lower() == "sped":
                    from dataloaders.val.SPEDDataset import SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=self.transform)
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )
            if self.show_data_stats:
                self.print_stats()


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
        result_dict = {}
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
                precision="float32",
            )

            self.log(
                f"{val_set_name}/R1",
                recalls_dict[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/R5",
                recalls_dict[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/R10",
                recalls_dict[10],
                prog_bar=False,
                logger=True,
            )
            result_dict[val_set_name] = recalls_dict
        
        return result_dict


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
