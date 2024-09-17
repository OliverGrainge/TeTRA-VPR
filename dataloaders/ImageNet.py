import os
import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from timm import create_model  # Using timm for Vision Transformer
import numpy as np
import torch.nn as nn

torch.set_float32_matmul_precision('medium')

class ImageNet(LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Define the transformations for training and validation
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.fc = nn.Linear(model.descriptor_dim, 1000)

    def forward(self, x):
        return self.fc(self.model(x))

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log(f"{prefix}_loss", loss_val, on_step=True, on_epoch=True)
        self.log(f"{prefix}_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_acc5", acc5, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cosine learning rate schedule with warmup
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs)
        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1.0, epoch / self.warmup_epochs))

        return [optimizer], [scheduler, warmup_scheduler]

    def process_batch(self, batch, transform):
        images = []
        for b in batch:
            img = np.array(b['image'])
            if img.ndim == 2 or img.shape[2] == 1:  # Grayscale image
                img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel image
            elif img.shape[2] > 3: 
                img = img[:, :, :3]
            
            images.append(transform(img))

        labels = [b['label'] for b in batch]
        return torch.stack(images), torch.tensor(labels)

    def train_dataloader(self):
        # Load the dataset using Hugging Face's datasets library (with streaming)
        train_dataset = load_dataset('imagenet-1k', split='train', streaming=False)

        # Apply transformations using a lambda function within the DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.workers, 
            shuffle=False,
            collate_fn=lambda batch: self.process_batch(batch, self.train_transforms)
        )
        return train_loader

    def val_dataloader(self):
        # Load the validation set
        val_dataset = load_dataset('imagenet-1k', split='validation', streaming=False)

        # Use the validation transformations
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.workers, 
            shuffle=False,
            collate_fn=lambda batch: self.process_batch(batch, self.val_transforms)
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    
if __name__ == '__main__':
    # Initialize the Vision Transformer model
    model = create_model('vit_base_patch16_224', pretrained=False)

    # Create the ViTLightningModel instance
    lit_model = ViTLightningModel(model=model, batch_size=32, workers=4, lr=3e-4, max_epochs=90)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="vit-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Create a trainer instance
    trainer = Trainer(
        max_epochs=90,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )

    # Start training
    trainer.fit(lit_model)
