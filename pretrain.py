import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("medium")


import argparse
import os

from config import DistillConfig, ModelConfig
from dataloaders.Distill import Distill


def setup_training(args):
    model_module = Distill(  #
        student_model_backbone_arch=args.backbone_arch,
        student_model_image_size=args.image_size,
        train_dataset_dir=args.train_dataset_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augmentation_level=args.augmentation_level,
        use_attn_loss=args.use_attn_loss,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        dirpath=f"./checkpoints/TeTRA-pretrain/Student[{model_module.student.name}]-Teacher[{model_module.teacher.name}]-Aug[{args.augmentation_level}]",
        filename="{epoch}-{step}-{train_loss:.4f}-{qfactor:.2f}",
        save_on_train_epoch_end=False,
        every_n_train_steps=250,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_top_k=3,
        mode="min",
    )

    final_checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/TeTRA-pretrain/Student[{model_module.student.name}]-Teacher[{model_module.teacher.name}]-Aug[{args.augmentation_level}]",
        filename="final-{epoch}-{step}-{train_loss:.4f}-{qfactor:.2f}",
        save_weights_only=True,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    learning_rate_cb = LearningRateMonitor(logging_interval="step")

    hyperparameters = {
        "student_model_backbone_arch": args.backbone_arch,
        "student_model_agg_arch": args.agg_arch,
        "student_model_image_size": args.image_size[0],
        "augmentation_level": args.augmentation_level,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "train_dataset_dir": args.train_dataset_dir,
        "max_epochs": args.max_epochs,
    }

    wandb_logger = WandbLogger(
        project="TeTRA-pretrain",
        name=f"Student[{model_module.student.name}]-Teacher[{model_module.teacher.name}]-Aug[{args.augmentation_level}]",
        config=hyperparameters,
    )

    trainer = pl.Trainer(
        enable_progress_bar=args.pbar,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, final_checkpoint_cb, learning_rate_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
        log_every_n_steps=50,
    )
    return trainer, model_module


if __name__ == "__main__":
    # Set precision for float32 matrix multiplication
    torch.set_float32_matmul_precision("medium")

    # Parse arguments
    parser = argparse.ArgumentParser()
    for config in [ModelConfig, DistillConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    # Setup and run training
    trainer, model_module = setup_training(args)
    trainer.fit(model_module)
