import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
import numpy as np
from PIL import Image
from collections import defaultdict
from matching.global_cosine_sim import global_cosine_sim
import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


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
        search_precision="float32",
        batch_size=32,
        num_workers=4,
        matching_function=global_cosine_sim,

    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.search_precision = search_precision
        self.matching_function = matching_function


    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import PittsburghDataset

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
                elif "sf_xl" in val_set_name.lower() and "val" in val_set_name.lower() and "small" in val_set_name.lower():
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(which_ds="sf_xl_small_val", input_transform=self.transform)
                    )
                elif "sf_xl" in val_set_name.lower() and "test" in val_set_name.lower() and "small" in val_set_name.lower():
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(which_ds="sf_xl_small_test", input_transform=self.transform)
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
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(value.detach().cpu())
        return descriptors["global_desc"].detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""

        results_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets): 
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            recalls_dict, _ = self.matching_function(**set_outputs, num_references=val_dataset.num_references, ground_truth=val_dataset.ground_truth)
            self.log_dict(
                recalls_dict,
                prog_bar=False,
                logger=True,
            )
            results_dict[val_set_name] = recalls_dict
        return results_dict
