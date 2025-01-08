import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")


import argparse
import os


from config import DataConfig, DistillConfig, ModelConfig
from dataloaders.Distill import Distill


def setup_training(args):
    model_module = Distill(  #
        student_model_backbone_arch=args.backbone_arch,
        student_model_agg_arch=args.agg_arch,
        student_model_image_size=args.image_size,
        teacher_model_preset=args.teacher_preset,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_set_names=args.val_set_names,
        use_attention=args.use_attention,
        use_progressive_quant=args.use_progressive_quant,
    )

    if args.use_progressive_quant:
        dirpath = f"./checkpoints/TeTRA-pretrain/{str(model_module.student)}-ProgressiveQuant"
    else:
        dirpath = f"./checkpoints/TeTRA-pretrain/{str(model_module.student)}"

    checkpoint_cb = ModelCheckpoint(
        monitor=f"{args.val_set_names[0]}_R1",
        dirpath=dirpath,
        filename="{epoch}-{" + args.val_set_names[0] + "_R1:.2f}",
        save_on_train_epoch_end=False,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_top_k=-1,
        mode="max",
    )

    learning_rate_cb = LearningRateMonitor(logging_interval="step")

    
    wandb_logger = WandbLogger(
        project="TeTRA-pretrain",
        name=f"{str(model_module.student)}",
    )

    trainer = pl.Trainer(
        enable_progress_bar=True,#args.pbar,
        devices=1,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb,learning_rate_cb],
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=0.5,#0.05,
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
