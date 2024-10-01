import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import pytorch_lightning as pl 
from pytorch_metric_learning import miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms as T

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

from models.helper import get_model

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

class QVPRDistill(pl.LightningModule):
    def __init__(
        self,
        config,
        teacher_arch="DinoSalad",
        student_backbone_arch="vit_small",
        student_agg_arch="cls",
        student_out_dim=1024,
        batch_size=32,
        image_size=(224, 224),
        num_workers=4,
        val_set_names=["pitts30k_val", "msls_val"],
        margin=0.1,
        max_epochs=10
    ):

        super().__init__()
        self.batch_size = (batch_size,)
        self.image_size = (image_size,)
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.max_epochs = max_epochs
    

        self.lr = config["lr"]
        self.optimizer_type = config["optimizer"]
        self.weight_decay = config["weight_decay"]
        self.momentum = config["momentum"]
        self.warmup_steps = config["warmup_steps"]
        self.milestones = config["milestones"]
        self.lr_mult = config["lr_mult"]
        self.miner_margin = config["miner_margin"]
        self.faiss_gpu = config["faiss_gpu"]
        self.img_per_place = config["img_per_place"]
        self.min_img_per_place = config["min_img_per_place"]
        self.cities = config["cities"]
        self.shuffle_all = config["shuffle_all"]


        # Data parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = IMAGENET_MEAN_STD["mean"]
        self.std_dataset = IMAGENET_MEAN_STD["std"]
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.show_data_stats = True
        self.warmup_epochs=1

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

        self.miner = miners.MultiSimilarityMiner(
            epsilon=margin, distance=CosineSimilarity()
        )
        self.teacher = get_model(preset=teacher_arch)
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            normalize_output=True,
            out_dim=student_out_dim,
        )

        self.freeze_model(self.teacher)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    @staticmethod
    def cosine_sim_vec(a, b):
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
        return torch.diag(a @ b.t())

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        student_desc = self.student(images)
        teacher_desc = self.teacher(images)
        miner_outputs = self.miner(teacher_desc, labels)
        a1, p, a2, n = miner_outputs
        teacher_pos_rel = self.cosine_sim_vec(teacher_desc[a1], teacher_desc[p])
        teacher_neg_rel = self.cosine_sim_vec(teacher_desc[a2], teacher_desc[n])
        student_pos_rel = self.cosine_sim_vec(student_desc[a1], student_desc[p])
        student_neg_rel = self.cosine_sim_vec(student_desc[a2], student_desc[n])
        teacher_rel = torch.cat((teacher_pos_rel, teacher_neg_rel))
        student_rel = torch.cat((student_pos_rel, student_neg_rel))

        loss = F.mse_loss(student_rel, teacher_rel)
        self.log("loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Compute total steps
        steps_per_epoch = len(self.train_dataloader())
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        # raise Exception
        # Scheduler

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

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

            recalls_dict_float, predictions = utils.get_validation_recalls(
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
                f"{val_set_name}/float_R1",
                recalls_dict_float[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R5",
                recalls_dict_float[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R10",
                recalls_dict_float[10],
                prog_bar=False,
                logger=True,
            )

            recalls_dict_binary, predictions = utils.get_validation_recalls(
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
                f"{val_set_name}/binary_R1",
                recalls_dict_binary[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R5",
                recalls_dict_binary[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R10",
                recalls_dict_binary[10],
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
