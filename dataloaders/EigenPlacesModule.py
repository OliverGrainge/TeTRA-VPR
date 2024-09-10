
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import utils
import time
torch.set_float32_matmul_precision('medium')


from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper
import os
import yaml
import argparse
import torch.nn as nn

from parsers import model_arguments, training_arguments, dataloader_arguments

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
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
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
    


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self,
                group_lengths,

                #---- Backbone
                backbone_arch=config['Model']['backbone_arch'],
                backbone_config=config['Model']['backbone_config'],
                
                #---- Aggregator
                agg_arch=config['Model']['agg_arch'], #CosPlace, NetVLAD, GeM, AVG
                agg_config=config['Model']['agg_config'],
                
                #---- Train hyperparameters
                lr=config['Training']['lr'],
                optimizer=config['Training']['optimizer'],
                weight_decay=config['Training']['weight_decay'],
                momentum=config['Training']['momentum'],
                warmpup_steps=config['Training']['warmup_steps'],
                milestones=config['Training']['milestones'],
                lr_mult=config['Training']['lr_mult'],
                
                #----- Loss
                loss_name=config['Training']['loss_name'],
                miner_name=config['Training']['miner_name'],
                miner_margin=config['Training']['miner_margin'],
                faiss_gpu=config['Training']['faiss_gpu'],
                search_precision=config['Training']['search_precision'], 

                lambda_lat=config['Training']['lambda_lat'],
                lambda_front=config['Training']['lambda_front'],
                groups_num=config['Training']['groups_num'],
                classifiers_lr = 0.01,

                 ):
        super().__init__()
        self.backbone_arch = backbone_arch
        self.backbone_config = backbone_config
        self.automatic_optimization = False

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        self.search_precision = search_precision

        self.lambda_lat = lambda_lat
        self.lambda_front = lambda_front
        self.classifiers_lr = classifiers_lr
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.classifiers = [MarginCosineProduct(
                self.output_dim, group_len, s=self.s, m=self.m) for group_len in self.groups_lengths]
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        model_opt = torch.optim.Adam([self.backbone.parameters()] + [self.aggregator.parameters()], lr=self.lr)
        classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=self.classifiers_lr) for classifier in self.classifiers]
        opt = [model_opt] + classifiers_optimizers
        return opt
    
    
    # This is the training step that's executed at each iteration
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
        dm = self.trainer.datamodule

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(dm.val_datasets) == 1:
            val_step_outputs = [self.validation_outputs]
        else:
            val_step_outputs = self.validation_outputs

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries    
            r_list = feats[:num_references]
            q_list = feats[num_references:]
        



            recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision=self.search_precision
            )

            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', recalls_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls_dict[10], prog_bar=False, logger=True)

            del r_list, q_list, feats, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs.clear()
        print('\n\n')
