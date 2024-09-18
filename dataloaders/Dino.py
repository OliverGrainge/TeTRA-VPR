import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from timm import create_model  # Using timm for Vision Transformer
from timm.data import create_transform
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from datasets import load_dataset

torch.set_float32_matmul_precision("medium")

class Dino(LightningModule):
    """
    PyTorch Lightning Model for training a Vision Transformer on ImageNet with Hugging Face Datasets.
    """

    def __init__(
        self,
        model,
        lr: float = 3e-4,  # Smaller learning rate for ViT
        weight_decay: float = 0.1,  # Higher weight decay for ViT
        batch_size: int = 32,
        workers: int = 4,
        warmup_epochs: int = 5,
        max_epochs: int = 90,
        **kwargs,
    ):
        
        self.model = model 
        self.learner = Dino(
            model,
            image_size = 224,
            hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,      # projector network hidden dimension
            projection_layers = 4,             # number of layers in projection network
            num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,                # student temperature
            teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
            global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        )


    def training_step(self, batch, batch_idx):
        images = batch 
        loss = self.learner(images)
        self.learner.update_moving_average()
        return loss