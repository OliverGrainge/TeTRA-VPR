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
from models.helper import get_model

def setup_training(args):
    student_model = get_model(
        backbone_arch=args.backbone_arch,
        agg_arch=args.agg_arch,
        image_size=args.image_size,
    )

    teacher_model = get_model(
        preset="DinoSalad",
    )

    model_module = Distill(
        student_model = student_model,
        teacher_model = teacher_model,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_set_names=args.val_set_names,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"{args.val_set_names[0]}_R1",
        dirpath=f"./checkpoints/TeTRA-pretrain/{str(student_model)}",
        filename="{epoch}-{" + args.val_set_names[0] + "_R1:.2f}",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="TeTRA-pretrain",
        name=f"{str(student_model)}",
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
