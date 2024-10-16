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
from torch.utils.data import Dataset
from PIL import Image
import tarfile
from io import BytesIO
import json
import glob
import os

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

from models.helper import get_model

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_feature_dim(model, image_size):
    x = torch.randn(3, *(image_size)).to(next(model.parameters()).device)
    features = model(x[None, :])
    return features.shape[1]

def freeze_model(model): 
    for param in model.parameters():
        param.requires_grad = False
    return model


class TarImageDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        tar_paths = glob.glob(os.path.join(data_directory, "*.tar"))
        self.tar_paths = tar_paths
        self.transform = transform
        self.image_paths = []
        
        # Store tar file info and image paths for later access
        self.tar_info = []
        for tar_path in tar_paths:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                self.tar_info.extend([(tar_path, member) for member in members if member.isfile()])
                
    def __len__(self):
        return len(self.tar_info)
    
    def __getitem__(self, idx):
        tar_path, member = self.tar_info[idx]
        with tarfile.open(tar_path, 'r') as tar:
            file = tar.extractfile(member)
            image = Image.open(BytesIO(file.read()))
            image = image.convert('RGB')  # Convert to RGB if necessary

        if self.transform:
            image = self.transform(image)
        
        return image
    

class ImageDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        self.image_paths = glob.glob(os.path.join(data_directory, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)    
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image
        
        


class VPRDistill(pl.LightningModule):
    def __init__(
        self,
        config,
        args,
        teacher_arch="DinoSalad",
        student_backbone_arch="vit_small",
        student_agg_arch="cls",
    ):
        super().__init__()
        self.teacher = get_model(preset=config["teacher_preset"])   
        self.student = get_model(backbone_arch=student_backbone_arch, agg_arch=student_agg_arch, out_dim=get_feature_dim(self.teacher, args.image_size))
        
        freeze_model(self.teacher)

        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = args.image_size
        self.lr = config["lr"]
        self.data_directory = config["data_directory"]
    
    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        images = batch 
        teacher_features = self.teacher(images)
        student_features = self(images)
        teacher_features = F.normalize(teacher_features, dim=1)
        student_features = F.normalize(student_features, dim=1)
        mse_loss = F.mse_loss(teacher_features, student_features)
        cosine_loss = (1 - F.cosine_similarity(teacher_features, student_features)).mean()
        loss = 1000 * mse_loss + cosine_loss
        print(loss.item(), 1000 * mse_loss.item(), cosine_loss.item())
        self.log("train_loss", loss)
        self.log("mse_loss", mse_loss)
        self.log("cosine_loss", cosine_loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr)
        return optimizer
    
    def train_transform(self):
        return T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"])
        ])
    
    def train_dataloader(self):
        paths = os.listdir(self.data_directory)
        if not os.path.isdir(self.data_directory):
            raise ValueError(f"Invalid data directory: {self.data_directory}")

        if "*tar" in paths[0]:
            train_dataset = TarImageDataset(self.data_directory, transform=self.train_transform())
        else:
            train_dataset = ImageDataset(self.data_directory, transform=self.train_transform())

        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)