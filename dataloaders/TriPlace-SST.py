import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()
        layers = [nn.Linear(in_dim, in_dim), nn.GELU()]
        if use_bn:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = self.mlp[-1]
        if norm_last_layer:
            self.last_layer.weight_g = nn.Parameter(torch.ones(out_dim))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, student_temp, teacher_temp, center_momentum):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
    
    def forward(self, student_outputs, teacher_outputs, center):
        total_loss = 0
        n_loss_terms = 0
        for student_output in student_outputs:
            for teacher_output in teacher_outputs:
                loss = self.compute_loss(student_output, teacher_output, center)
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
    
    def compute_loss(self, student_output, teacher_output, center):
        teacher_output = teacher_output.detach()
        teacher_output = nn.functional.softmax((teacher_output - center) / self.teacher_temp, dim=-1)
        student_output = nn.functional.log_softmax(student_output / self.student_temp, dim=-1)
        loss = torch.sum(-teacher_output * student_output, dim=-1).mean()
        return loss
    


class DataAugmentationDINO:
    def __init__(self, config):
        self.global_crop_scale = config['global_crop_scale']
        self.local_crop_scale = config['local_crop_scale']
        self.global_crops_number = config['global_crops_number']
        self.local_crops_number = config['local_crops_number']
        
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            )], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])
        
        # Global crops
        self.global_transforms = transforms.Compose([
            transforms.RandomResizedCrop(config['image_size'], scale=self.global_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(1.0),
            normalize,
        ])
        
        # Local crops
        self.local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(config['image_size'], scale=self.local_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(p=0.1),
            transforms.RandomSolarize(170, p=0.2),
            normalize,
        ])
        
        # List of transformations
        self.transforms = [self.global_transforms] * self.global_crops_number + [self.local_transforms] * self.local_crops_number
    
    def __len__(self):
        return len(self.transforms)
    
    def __getitem__(self, idx):
        return self.transforms[idx]
    


class DINOTrainer(pl.LightningModule):
    def __init__(self, student_model, teacher_model, config):
        super(DINOTrainer, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.config = config
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Create projection heads if necessary
        self.student_head = DINOHead(self.config['embed_dim'], self.config['out_dim'])
        self.teacher_head = DINOHead(self.config['embed_dim'], self.config['out_dim'])
        
        # Temperature parameters
        self.student_temp = self.config['student_temp']
        self.teacher_temp = self.config['teacher_temp']
        self.center_momentum = self.config['center_momentum']
        
        # Centering for teacher outputs
        self.register_buffer("center", torch.zeros(1, self.config['out_dim']))
        
        # Loss function
        self.criterion = DINOLoss(self.student_temp, self.teacher_temp, self.center_momentum)
        
        # Data augmentations
        self.train_transforms = DataAugmentationDINO(self.config)
    
    def forward(self, x):
        # Forward pass through student model
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch  # Unlabeled data
        
        # Generate views
        views = [transform(images) for transform in self.train_transforms]
        student_outputs = []
        with torch.no_grad():
            teacher_outputs = []
            for view in views:
                view = view.to(self.device)
                teacher_output = self.teacher(view)
                teacher_output = self.teacher_head(teacher_output)
                teacher_outputs.append(teacher_output)
        
        for view in views:
            view = view.to(self.device)
            student_output = self.student(view)
            student_output = self.student_head(student_output)
            student_outputs.append(student_output)
        
        # Compute loss
        loss = self.criterion(student_outputs, teacher_outputs, self.center)
        
        # Update center
        self.update_center(teacher_outputs)
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.config['learning_rate'])
        return optimizer
    
    @torch.no_grad()
    def update_center(self, teacher_outputs):
        batch_center = torch.mean(torch.cat(teacher_outputs), dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)