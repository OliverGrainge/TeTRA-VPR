import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")


import argparse
import os

import yaml

from dataloaders.Distill import Distill
from dataloaders.TeTRA import TeTRA
from models.helper import get_model
from config import DataConfig, ModelConfig, DistillConfig, TeTRAConfig

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


def load_model(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        out_dim=args.out_dim,
        normalize_output=True,
    )

    if not os.path.exists(args.load_checkpoint):
        raise ValueError(f"Checkpoint {args.weights_path} does not exist")
        
    sd = torch.load(args.weights_path)["state_dict"]
    backbone_sd = {
        k.replace("backbone.", ""): v
        for k, v in sd.items()
        if k.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone_sd, strict=False)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.freeze()
    
    return model

def setup_training(args, model):
    model_module = TeTRA(
        model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_set_names=args.val_set_names,
        loss_name=args.loss_name,
        miner_name=args.miner_name,
        miner_margin=0.1,
        cities=args.cities,
        lr=args.lr,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="global_cosine_sim_{args.val_set_names[0]}_R1",
        dirpath=_get_checkpoint_dir(args),
        filename=_get_checkpoint_filename(args),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project=args.training_method.lower(),
        name=_get_wandb_run_name(args),
    )

    trainer = pl.Trainer(
        enable_progress_bar=True,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
    )
    
    return trainer, model_module


def _get_checkpoint_dir(args):
    return f"./checkpoints/TeTRA/backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]",

def _get_checkpoint_filename(args):
    return (f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_aug[{args.augment_level.lower()}]"
            f"_loss_name[{args.loss_name}]_miner_name[{args.miner_name}]]_res[{args.image_size[0]}x{args.image_size[1]}]"
            "_epoch({epoch:02d})_step({step:04d})_R1({global_cosine_sim_{args.val_set_names[0]}_R1:.4f})")

def _get_wandb_run_name(args):
    return f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_dim[{args.out_dim}]_lossname[{args.loss_name}]_minername[{args.miner_name}]_res[{args.image_size[0]}x{args.image_size[1]}]"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, TeTRAConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    model = load_model(args)
    trainer, model_module = setup_training(args, model)
    trainer.fit(model_module)
