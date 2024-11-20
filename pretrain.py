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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataConfig.add_argparse_args(parser)
    parser = ModelConfig.add_argparse_args(parser)
    parser = DistillConfig.add_argparse_args(parser)
    args = parser.parse_args()

    
    model_module = Distill(
        data_directory=args.train_dataset_dir,
        student_backbone_arch=args.backbone_arch,
        student_agg_arch=args.agg_arch,
        teacher_preset=args.teacher_preset,
        lr=args.lr,
        mse_loss_mult=args.mse_loss_mult,
        batch_size=args.batch_size,
        weight_decay_init=args.weight_decay_init,
        weight_decay_schedule=args.weight_decay_schedule,
        use_attention=args.use_attention,
        image_size=args.image_size,
        augment_level=args.augment_level,
        num_workers=args.num_workers,
        val_set_names=args.val_set_names,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"{args.val_set_names[0]}_R1",
        dirpath=f"./checkpoints/Distill/backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_teacher[{args.teacher_preset.lower()}]",
        filename=f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_teacher[{args.teacher_preset.lower()}]_res[{args.image_size[0]}x{args.image_size[1]}]_aug[{args.augment_level.lower()}]_decay[{args.weight_decay_schedule.lower()}]"
        + "_epoch({epoch:02d})_step({step:04d})_R1({pitts30k_val_R1:.4f})",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project=args.training_method.lower(),  # Replace with your project name
        name=f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_teacher[{args.teacher_preset.lower()}]_Res[{args.image_size[0]}x{args.image_size[1]}]_aug[{args.augment_level.lower()}]_decay[{args.weight_decay_schedule.lower()}]",
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
        val_check_interval=0.05,
        logger=wandb_logger,  # Add the wandb logger here
    )

    trainer.fit(model_module)
