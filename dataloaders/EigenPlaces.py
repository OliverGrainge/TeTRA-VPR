import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os
import yaml
from pytorch_lightning.trainer.supporters import CombinedLoader
from prettytable import PrettyTable
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.train.EigenPlacesDataset import EigenPlacesDataset
from val.PittsburghDataset import PittsburghDataset
from val.MapillaryDataset import MSLS
from val.NordlandDataset import NordlandDataset
from val.SPEDDataset import SPEDDataset
import utils
import helper

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
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

class EigenPlaces(pl.LightningModule):
    def __init__(self,
                 config,
                 model,
                 image_size=224, 
                 batch_size=32, 
                 val_set_names=['pitts30k_val'], 
                 output_dim=2048,
                 mean_std=IMAGENET_MEAN_STD,
                 num_workers=8, 
                 shuffle_all=True):
        super().__init__()

        # Backbone and Aggregator from config
        self.backbone_arch = config['Model']['backbone_arch']
        self.backbone_config = config['Model']['backbone_config']
        self.agg_arch = config['Model']['agg_arch']
        self.agg_config = config['Model']['agg_config']

        # Training hyperparameters
        self.lr = config['Training']['lr']
        self.optimizer_type = config['Training']['optimizer']
        self.weight_decay = config['Training']['weight_decay']
        self.momentum = config['Training']['momentum']
        self.warmup_steps = config['Training']['warmup_steps']
        self.milestones = config['Training']['milestones']
        self.lr_mult = config['Training']['lr_mult']

        # Loss and mining
        self.loss_name = config['Training']['loss_name']
        self.miner_name = config['Training']['miner_name']
        self.miner_margin = config['Training']['miner_margin']
        self.loss_fn = utils.get_loss(self.loss_name)
        self.miner = utils.get_miner(self.miner_name, self.miner_margin)
        self.batch_acc = []
        
        # FAISS settings
        self.faiss_gpu = config['Training']['faiss_gpu']
        self.search_precision = config['Training']['search_precision']
        
        # Lateral and Frontal Loss Scaling
        self.lambda_lat = config['Training']['lambda_lat']
        self.lambda_front = config['Training']['lambda_front']
        self.classifiers_lr = 0.01

        # Get backbone and aggregator
        self.model = model

        self.groups = [EigenPlacesDataset(
                M=self.M, N=self.N, focal_dist=self.focal_dist, current_group=n//2,
                min_images_per_class=self.min_images_per_class, angle=[0, 90][n % 2],
                visualize_classes=self.visualize_classes) for n in range(self.groups_num * 2)
            ]

        # Group-specific classifiers
        self.classifiers = [MarginCosineProduct(output_dim, len(group), s=s, m=m) for group in self.groups]

        # Data configuration
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_set_names = val_set_names
        self.output_dim = output_dim
        self.M = config["Training"]["M"]
        self.N = config["Training"]["N"]
        self.s = config["Training"]["s"]
        self.m = config["Training"]["m"]
        self.n = config["Training"]["n"]
        self.focal_dist = config["Training"]["focal_dist"]
        self.min_images_per_class = config["Training"]["M"]["min_images_per_class"]
        self.visualize_classes = config["Training"]["M"]["visualize_classes"]
        self.groups_num = self.N * self.N
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.num_workers = num_workers
        self.shuffle_all = shuffle_all

        # Train and valid transforms
        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])

        # Dataloader configs
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': False,
            'shuffle': self.shuffle_all
        }

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers // 2,
            'drop_last': False,
            'pin_memory': False,
            'shuffle': False
        }
        

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        model_opt = torch.optim.Adam([self.backbone.parameters()] + [self.aggregator.parameters()], lr=self.lr)
        classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=self.classifiers_lr) for classifier in self.classifiers]
        opt = [model_opt] + classifiers_optimizers
        return opt
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        opt = self.optimizers()
        for dataset_num, b in batch.items():
            images, targets, _ = b
            descriptors = self(images)
            output = self.classifiers[dataset_num](descriptors, targets)
            loss = self.criterion(output, targets)
            if dataset_num % 2 == 0:
                loss *= self.lambda_lat 
                self.log("lateral loss", loss)
            else: 
                loss *= self.lambda_front 
                self.log("frontal loss", loss)
            self.manual_backward(loss)
            opt[0].step()
            opt[dataset_num + 1].step()
            opt[0].zero_grad()
            opt[dataset_num + 1].zero_grad()

    def train_dataloader(self):
        # Dataloaders for different datasets in different groups
        loaders = {}
        for group_num in range(self.groups_num):
            loaders[group_num] = DataLoader(self.groups[group_num], **self.train_loader_config)
        return loaders

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders


