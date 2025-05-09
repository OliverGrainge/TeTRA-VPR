import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from config import ModelConfig, TeTRAConfig
from dataloaders.TeTRA import TeTRA
from models.helper import get_model
import torch.nn as nn


#BACKBONE_WEIGHT_PATH = "/home/oeg1n18/QuantPlaceFinder/checkpoints/TeTRA-pretrain/Student[VitbaseT322]-Teacher[DinoV2]-Aug[Severe]/epoch=17-step=138750-train_loss=0.0463-qfactor=1.00.ckpt"

def _freeze_module(module): 
    for param in module.parameters(): 
        param.requires_grad = False

def _unfreeze_module(module): 
    for param in module.parameters(): 
        param.requires_grad = True

def _freeze_tetra_backbone(model: nn.Module, unfreeze_n_last_layers: int = 1):
    backbone = model.backbone

    _freeze_module(backbone)

    # set dropout to 0 for all layers except the last unfreeze_n_last_layers
    for block in backbone.transformer.layers[:-unfreeze_n_last_layers]:
        for name, module in block.named_modules(): 
            if isinstance(module, nn.Dropout): 
                module.p = 0.0
                
    # only train the last unfreeze_n_last_layers
    for block in (backbone.transformer.layers[-unfreeze_n_last_layers:]):
        _unfreeze_module(block)
    
    # make sure the backbone is in fully quantized mode 
    for module in model.modules(): 
        if hasattr(module, "set_qfactor"):
            module.set_qfactor(1.0) 

    return model 


def _freeze_dino_backbone(model: nn.Module, unfreeze_n_last_layers: int = 3):
    _freeze_module(model.backbone.dino.patch_embed) 

    for blk in model.backbone.dino.blocks[:-unfreeze_n_last_layers]: 
        _freeze_module(blk)
        
    return model 

def _freeze_backbone(model: nn.Module, unfreeze_n_last_layers: int = 1):
    if model.name.lower().startswith("dino"): 
        return _freeze_dino_backbone(model, unfreeze_n_last_layers)
    else: 
        return _freeze_tetra_backbone(model, unfreeze_n_last_layers)

def _load_backbone_weights(model: nn.Module, backbone_weights_path: str):
    sd = torch.load(backbone_weights_path, weights_only=False)["state_dict"]
    new_sd = {}
    for key, value in sd.items(): 
        if key.startswith("student"): 
            new_sd[key.replace("student.", "")] = value 

    model.backbone.load_state_dict(new_sd)
    return model 



def load_model(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        desc_divider_factor=args.desc_divider_factor,
    )

    if args.backbone_checkpoint is not None: 
        model = _load_backbone_weights(model, backbone_weights_path=args.backbone_checkpoint)
        print("=================================================================")
        print("Loaded backbone weights from: ", args.backbone_checkpoint)
        print("=================================================================")

    if args.freeze_backbone: 
        model = _freeze_backbone(model, unfreeze_n_last_layers=1)
        print("=================================================================")
        print("Frozen backbone weights")
        print("=================================================================")
    else: 
        _unfreeze_module(model)
    model.train()
    return model



def setup_training(args, model):
    model_module = TeTRA(
        model,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        val_set_names=args.val_set_names,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        cities=args.cities,
        lr=args.lr,
        quant_schedule=args.quant_schedule,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"MSLS_binary_R1",  # msls_val_q_R1
        dirpath=f"./checkpoints/TeTRA-finetune/{model.name}-QSch[{args.quant_schedule}]-Frozen[{args.freeze_backbone}]",
        filename="{epoch}-{MSLS_binary_R1:.2f}",  # msls_val_q_R1
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="TeTRA-finetune",
        name=f"{model.name}-DD[{args.desc_divider_factor}]",
    )

    trainer = pl.Trainer(
        enable_progress_bar=args.pbar,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        #limit_train_batches=50,
    )

    return trainer, model_module


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    for config in [ModelConfig, TeTRAConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    model = load_model(args)
    trainer, model_module = setup_training(args, model)
    trainer.fit(model_module)


