import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")


import argparse
import os

import yaml

from config import DataConfig, DistillConfig, ModelConfig
from dataloaders.Distill import Distill



def setup_training(args):
    model_module = Distill(
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
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
        augmentation_level=args.augmentation_level,
        num_workers=args.num_workers,
        val_set_names=args.val_set_names,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"{args.val_set_names[0]}_R1",
        dirpath=_get_checkpoint_dir(args),
        filename="{epoch}-{" + args.val_set_names[0] + "_R1:.2f}",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="Distill",
        name=_get_wandb_run_name(args),
    )

    trainer = pl.Trainer(
        enable_progress_bar=args.pbar,
        devices=1, 
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=0.05,
        log_every_n_steps=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
    )

    return trainer, model_module


def _get_checkpoint_dir(args):
    return f"./checkpoints/Distill/backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_teacher[{args.teacher_preset.lower()}]_res[{args.image_size[0]}x{args.image_size[1]}]_aug[{args.augmentation_level.lower()}]_decay[{args.weight_decay_schedule.lower()}]"


def _get_wandb_run_name(args):
    return f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_teacher[{args.teacher_preset.lower()}]_res[{args.image_size[0]}x{args.image_size[1]}]_aug[{args.augmentation_level.lower()}]_decay[{args.weight_decay_schedule.lower()}]"


if __name__ == "__main__":
    # Set precision for float32 matrix multiplication
    torch.set_float32_matmul_precision("medium")

    # Parse arguments
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, DistillConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    # Setup and run training
    trainer, model_module = setup_training(args)
    trainer.fit(model_module)
