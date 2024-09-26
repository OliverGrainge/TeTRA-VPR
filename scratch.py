import torch
import copy
import random
from functools import wraps, partial
from tqdm import tqdm 
import torch
from torch import nn
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import transforms as T
from vit_pytorch import ViT
import os 
from glob import glob 
import tarfile

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss function # (algorithm 1 in the paper)

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.flatten(1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        embed = self.get_embedding(x)
        if not return_projection:
            return embed

        projector = self._get_projector(embed)
        return projector(embed), embed



# main class

class Dino(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.8,#0.4,
        global_lower_crop_scale = 0.8,#0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None,
        teacher_arch = 'dinov2_vitb14',
    ):
        super().__init__()
        self.net = net
        self.teacher_arch = teacher_arch
        # default BYOL augmentation

        augments = T.Compose([
            #T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #RandomApply(
            #    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            #    p = 0.3
            #),
            #T.RandomGrayscale(p=0.1),
            #T.RandomRotation(degrees=15),
            #T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            #T.RandomHorizontalFlip(),
            #T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
            #RandomApply(
            #    T.GaussianBlur((3, 3), (0.5, 1.0)),
            #    p = 0.2
            #),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        ])

        self.panorama_crop = T.Compose([
            #transforms.ToTensor(),
            T.RandomCrop((512, 512)),  
            T.Resize((image_size, image_size))    
        ])

        self.augment1 = augments
        self.augment2 = augments



        # local and global crops
        print("==============", 0.6, local_upper_crop_scale)
        self.local_crop = T.Compose(
            [#T.RandomRotation(degrees=10), #T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            T.RandomResizedCrop((image_size, image_size), scale = (0.6, local_upper_crop_scale))]
        )
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))
        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = 'to_latent')

        teacher_net = torch.hub.load('facebookresearch/dinov2', self.teacher_arch)
        set_requires_grad(teacher_net, False)

        self.teacher_encoder = NetWrapper(teacher_net, num_classes_K, projection_hidden_size, projection_layers, layer = 'head')
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, 512, 1024, device=device))


    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        # when distilling from teacher the below line is not required
        #update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        #print("=========", x.shape)
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)
        
        #axes[5].imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        #axes[5].set_title("panorama")
        #plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        #plt.show(block=False)
        x = self.panorama_crop(x)
        #print("===", x.shape)
        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        """
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        axes = axes.flatten()
        axes[0].imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[0].set_title("panorama crop")
        axes[1].imshow(image_one[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[1].set_title("image_one")
        axes[2].imshow(image_two[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[2].set_title("image_two")
        axes[3].imshow(local_image_one[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[3].set_title("local_image_one")
        axes[4].imshow(global_image_one[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[4].set_title("global_image_one")
        

        plt.show()
        """
        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        with torch.no_grad():
            teacher_proj_one, _ = self.teacher_encoder(global_image_one)
            teacher_proj_two, _ = self.teacher_encoder(global_image_two)

        loss_fn_ = partial(
            loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        dino_loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2

        return dino_loss



import pytorch_lightning as pl 
import matplotlib.pyplot as plt

class DinoSSL(pl.LightningModule): 
    def __init__(self, model, dataset_dir="", lr=1e-3, batch_size=64, num_workers=4, teacher_arch='dinov2_vitb14', image_size=224, hidden_layer=-2, projection_hidden_size=224, projection_layers=4, num_classes_K = 65336, student_temp=0.9, teacher_temp=0.04, 
                 local_upper_crop_scale=0.8, global_lower_crop_scale=0.8, moving_average_decay=0.9, center_moving_average_decay=0.9): 
        super().__init__()
        self.model = model
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size 
        self.num_workers = num_workers
        self.teacher_arch = teacher_arch
        self.lr = lr
        self.total_images = self.count_total_images(os.path.join(self.dataset_dir, "shard_*.tar"))

        self.learner = Dino(
            self.model, 
            image_size=image_size, 
            teacher_arch = self.teacher_arch,
            hidden_layer=hidden_layer, 
            projection_hidden_size=projection_hidden_size, 
            projection_layers=projection_layers, 
            num_classes_K=num_classes_K, 
            student_temp=student_temp, 
            local_upper_crop_scale=local_upper_crop_scale, 
            global_lower_crop_scale=global_lower_crop_scale, 
            moving_average_decay=moving_average_decay, 
            center_moving_average_decay=center_moving_average_decay,
        )


    def training_step(self, batch, batch_idx):
        images = batch[0] 
        #img = images[0]
        #img = img.permute(1, 2, 0).detach().cpu().numpy()
        #plt.imshow(img)
        #plt.show()
        loss = self.learner(images)
        self.log("train_loss", loss)
        return loss 

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.learner.update_moving_average()

    def count_total_images(self, shard_pattern):
        total = 0
        for shard in glob(shard_pattern):
            with tarfile.open(shard, "r") as tar:
                total += len([member for member in tar.getmembers() if member.isfile()])
        return total

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor()  # Converts PIL Image to Tensor
        ])
        n_shards = len(os.listdir(self.dataset_dir)) - 1
        shards = os.path.join(self.dataset_dir, "shard_{000000.."+ f"{n_shards:06d}" + "}.tar")
        dataset = wds.WebDataset(shards).decode("pil").to_tuple("jpg").map(lambda x: (transform(x[0]),))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return dataloader

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.learner.parameters(), lr=self.lr)
        return opt



model = ViT(
    image_size = 224,   # Standard image size for ViT.
    patch_size = 16,    # Standard patch size.
    num_classes = 1000, # Number of classes for the dataset (ImageNet or similar).
    dim = 368,          # Embedding dimension, which can be reduced for a smaller ViT.
    depth = 6,          # Transformer layers, suitable for a small model.
    heads = 8,          # Number of attention heads, aligned with the embedding dimension.
    mlp_dim = 2048      # MLP dimension, related to the transformer depth and heads.
)


module = DinoSSL(model, dataset_dir='/home/oliver/datasets_drive/vpr_datasets/sf_xl/compressed/dataset')

trainer = pl.Trainer(
        enable_progress_bar=True,
        strategy="auto",
        accelerator='cuda',
        precision='bf16-mixed',
        max_epochs=100,
        limit_train_batches=module.total_images//module.batch_size
    )





trainer.fit(module)



"""

learner = Dino(
    model,
    image_size = 224,
    hidden_layer = -2,        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 224,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)


opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

def sample_unlabelled_images():
    return torch.randn(1, 3, 224, 224)

for _ in tqdm(range(100)):
    images = sample_unlabelled_images()
    #images = images.to("cuda")
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers
"""




