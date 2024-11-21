from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from tabulate import tabulate


from matching.match_cosine import match_cosine


def get_sample_output(model, transform):
    model.eval()
    sample = np.random.randint(0, 255, size=(224, 224, 3))
    sample = Image.fromarray(sample.astype(np.uint8))
    sample = transform(sample)
    sample = sample.unsqueeze(0)
    output = model(sample)
    return output


class VPREval(pl.LightningModule):
    def __init__(
        self,
        model,
        transform,
        val_set_names=["pitts30k"],
        val_dataset_dir=None, 
        batch_size=32,
        num_workers=4,
        matching_function=match_cosine,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.matching_function = matching_function
        self.val_dataset_dir = val_dataset_dir

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset30k
                    self.val_datasets.append(
                        PittsburghDataset30k(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "pitts250k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset250k
                    self.val_datasets.append(
                        PittsburghDataset250k(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "msls" in val_set_name.lower():
                    from dataloaders.val.MapillaryDataset import MSLS
                    self.val_datasets.append(
                        MSLS(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "nordland" in val_set_name.lower():
                    from dataloaders.val.NordlandDataset import NordlandDataset
                    self.val_datasets.append(
                        NordlandDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "sped" in val_set_name.lower():
                    from dataloaders.val.SPEDDataset import SPEDDataset
                    self.val_datasets.append(
                        SPEDDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test"))
                elif "essex" in val_set_name.lower():
                    from dataloaders.val.EssexDataset import EssexDataset
                    self.val_datasets.append(
                        EssexDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "sanfrancicscosmall" in val_set_name.lower():
                    from dataloaders.val.SanFranciscoSmall import SanFranciscoSmall
                    self.val_datasets.append(
                        SanFranciscoSmall(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "tokyo" in val_set_name.lower():
                    from dataloaders.val.Tokyo247 import Tokyo247
                    self.val_datasets.append(
                        Tokyo247(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                elif "cross" in val_set_name.lower():
                    from dataloaders.val.CrossSeasonDataset import CrossSeasonDataset
                    self.val_datasets.append(
                        CrossSeasonDataset(val_dataset_dir=self.val_dataset_dir, input_transform=self.transform, which_set="test")
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
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

    @torch.no_grad()
    def forward(self, x):
        x = self.model(x)
        return x

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
        self.results = {}
        results_dict = {}

        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _, _ = self.matching_function(
                **set_outputs,
                num_references=val_dataset.num_references,
                ground_truth=val_dataset.ground_truth,
            )

            # Store results in the dictionary
            results_dict[val_dataset.__repr__()] = recalls_dict

        # Return the results to the Trainer
        self.results = results_dict 
        return None


