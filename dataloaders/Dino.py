import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# Custom Solarize Transformation (if needed)
class RandomSolarize:
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return torch.where(img > self.threshold / 255, 1 - img, img)
        return img


class Dino(LightningModule):
    """
    PyTorch Lightning Module for training a Vision Transformer using the DINO method.
    Accepts a single nn.Module object and internally creates student and teacher models.
    """

    def __init__(
        self,
        base_model: nn.Module,  # Single model instance
        image_size: int = 224,
        projection_hidden_size: int = 256,
        projection_layers: int = 3,
        num_classes_K: int = 65536,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        momentum_teacher: float = 0.996,
        weight_decay: float = 0.04,
        lr: float = 0.0005,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        batch_size: int = 32,
        workers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Student and Teacher models
        self.student = base_model
        self.teacher = copy.deepcopy(base_model)
        self._initialize_teacher()

        # Projection heads for both networks
        self.student_proj = self._build_projection_head()
        self.teacher_proj = self._build_projection_head()

        # Freeze teacher parameters and projection head
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_proj.parameters():
            param.requires_grad = False

        # DINO specific parameters
        self.temperature_student = student_temp
        self.temperature_teacher = teacher_temp
        self.momentum_teacher = momentum_teacher

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Data augmentation
        self.train_transform = self._build_transforms()

    def _build_projection_head(self):
        """
        Builds the projection head as per DINO.
        """
        layers = []
        # Determine the input dimension from the base model
        # Adjust the attribute based on your model's architecture
        input_dim = self.student.descriptor_dim

        for _ in range(self.hparams.projection_layers - 1):
            layers.append(nn.Linear(input_dim, self.hparams.projection_hidden_size))
            layers.append(nn.BatchNorm1d(self.hparams.projection_hidden_size))
            layers.append(nn.ReLU())
            input_dim = self.hparams.projection_hidden_size
        layers.append(nn.Linear(input_dim, self.hparams.num_classes_K))
        layers.append(nn.BatchNorm1d(self.hparams.num_classes_K, affine=False))
        return nn.Sequential(*layers)

    def _initialize_teacher(self):
        """
        Initializes the teacher model with the student model's weights.
        """
        for param_student, param_teacher in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_teacher.data.copy_(param_student.data)
        # If using separate projection heads, initialize teacher's projection head
        # Assuming teacher_proj is already defined
        if hasattr(self, "student_proj") and hasattr(self, "teacher_proj"):
            for param_student_proj, param_teacher_proj in zip(
                self.student_proj.parameters(), self.teacher_proj.parameters()
            ):
                param_teacher_proj.data.copy_(param_student_proj.data)

    @torch.no_grad()
    def update_teacher(self):
        """
        Updates the teacher model parameters using an exponential moving average of the student model parameters.
        """
        for param_student, param_teacher in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_teacher.data = (
                param_teacher.data * self.momentum_teacher
                + param_student.data * (1.0 - self.momentum_teacher)
            )
        for param_student_proj, param_teacher_proj in zip(
            self.student_proj.parameters(), self.teacher_proj.parameters()
        ):
            param_teacher_proj.data = (
                param_teacher_proj.data * self.momentum_teacher
                + param_student_proj.data * (1.0 - self.momentum_teacher)
            )

    def _build_transforms(self):
        """
        Builds the data augmentation pipeline for training.
        """
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.hparams.image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(
                    kernel_size=23, sigma=(0.1, 2.0)
                ),  # Added Gaussian Blur
                RandomSolarize(threshold=128, p=0.2),  # Custom Solarize
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)  # Optional
            ]
        )

    def forward_student(self, x):
        """
        Forward pass through the student network.
        """
        features = self.student(x)
        projections = self.student_proj(features)
        return projections

    def forward_teacher(self, x):
        """
        Forward pass through the teacher network.
        """
        with torch.no_grad():
            features = self.teacher(x)
            projections = self.teacher_proj(features)
        return projections

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        Expects batch to be a tuple of two augmented views.
        """
        (x1, x2) = batch  # Two augmented views

        # Student forward pass
        student_out_1 = self.student_proj(self.student(x1))  # First view
        student_out_2 = self.student_proj(self.student(x2))  # Second view

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_out_1 = self.teacher_proj(self.teacher(x1))  # First view
            teacher_out_2 = self.teacher_proj(self.teacher(x2))  # Second view

        # Normalize outputs
        student_out_1 = F.log_softmax(student_out_1 / self.temperature_student, dim=-1)
        student_out_2 = F.log_softmax(student_out_2 / self.temperature_student, dim=-1)
        teacher_out_1 = F.softmax(teacher_out_1 / self.temperature_teacher, dim=-1)
        teacher_out_2 = F.softmax(teacher_out_2 / self.temperature_teacher, dim=-1)

        # Compute loss
        loss_1 = self.criterion(student_out_1, teacher_out_2)
        loss_2 = self.criterion(student_out_2, teacher_out_1)
        loss = (loss_1 + loss_2) / 2

        # Log loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Update teacher
        self.update_teacher()

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(
            list(self.student.parameters()) + list(self.student_proj.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        return [optimizer], [scheduler]

    def train_dataloader(self):
        """
        Define your training dataloader here.
        Ensures that it returns two augmented views of each image.
        """

        class TwoCropsTransform:
            """
            Creates two augmented versions of each image.
            """

            def __init__(self, transform):
                self.transform = transform

            def __call__(self, x):
                return self.transform(x), self.transform(x)

        transform = TwoCropsTransform(self.train_transform)

        dataset = ImageFolder(root="path_to_imagenet/train", transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers,
            pin_memory=True,
            drop_last=True,
        )
