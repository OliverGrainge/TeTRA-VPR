import pytorch_lightning as pl
from torch.utils.data import DataLoader
from train.EigenPlacesDataset import EigenPlacesDataset
from torchvision import transforms as T
import os 
from val.PittsburghDataset import PittsburghDataset
from val.MapillaryDataset import MSLS
from val.NordlandDataset import NordlandDataset
from val.SPEDDataset import SPEDDataset
import torch 
import torch.nn as nn
from torch.nn import Parameter
from typing import Type
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader



def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

class EigenPlacesDataModule(pl.LightningDataModule):

    def __init__(self,
                 image_size=224, 
                 batch_size=32, 
                 val_set_names=['pitts30k_val'], 
                 output_dim=2048, 
                 M=15,
                 N=3,
                 s=100, 
                 m=0.4, 
                 n=2,#
                 focal_dist=25, 
                 min_images_per_class=5, 
                 visualize_classes=0,
                 mean_std=IMAGENET_MEAN_STD,
                 num_workers=8, 
                 shuffle_all=True):
        
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_set_names = val_set_names
        self.output_dim = output_dim 
        self.M = M 
        self.N = N 
        self.s = s 
        self.m = m 
        self.n = n
        self.focal_dist = focal_dist 
        self.min_images_per_class = min_images_per_class 
        self.visualize_classes = visualize_classes
        self.groups_num = N * N
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.num_workers = num_workers
        self.shuffle_all = shuffle_all

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])
        
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': True,
            'pin_memory': False,
            'shuffle': self.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': False,
            'shuffle': False}
        
        self.epoch = 0 

    def setup(self, stage=None):
        # Setup datasets (train, val, test) based on the stage: 'fit', 'test', etc.
        if stage == 'fit' or stage is None:
            self.epoch = 0
            self.groups = [
                EigenPlacesDataset(
                    M=self.M, N=self.N, 
                    focal_dist=self.focal_dist, current_group=n//2, 
                    min_images_per_class=self.min_images_per_class, 
                    angle=[0, 90][n % 2], visualize_classes=self.visualize_classes
                ) for n in range(self.groups_num * 2)
            ]

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if 'pitts30k' in valid_set_name.lower():
                    self.val_datasets.append(PittsburghDataset(which_ds=valid_set_name,
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'msls_val':
                    self.val_datasets.append(MSLS(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'nordland':
                    self.val_datasets.append(NordlandDataset(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'sped':
                    self.val_datasets.append(SPEDDataset(
                        input_transform=self.valid_transform))
                else:
                    print(
                        f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
                
    
    def train_dataloader(self):
        current_dataset_num = (self.epoch % self.groups_num) * 2
        loaders = {}
        for i in range(2):
            loaders[current_dataset_num + 1] = DataLoader(self.groups[current_dataset_num + i], **self.train_loader_config) 
        return loaders

    def val_dataloader(self):
        val_dataloaders = []

        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders
    




if __name__ == "__main__": 
    dl = EigenPlacesDataModule()
    dl.setup('fit')
