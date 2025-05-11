import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("medium")


import argparse
import os

from config import DistillConfig, ModelConfig
from dataloaders.Distill import Distill


def _argparse(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="myyaml.yaml")
    return parser.parse_args()

def setup_training(conf):
    model_module = Distill(  #
        student_model_backbone_arch=conf.backbone_arch,
        student_model_image_size=conf.image_size,
        train_dataset_dir=conf.train_dataset_dir,
        lr=conf.lr,
        batch_size=conf.batch_size,
        weight_decay=conf.weight_decay,
        image_size=conf.image_size,
        num_workers=conf.num_workers,
        augmentation_level=conf.augmentation_level,
        use_attn_loss=conf.use_attn_loss,
    )
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        dirpath=f"./checkpoints/TeTRA-pretrain/{config_name}",
        filename="{epoch}-{train_loss:.4f}-{qfactor:.2f}",
        save_on_train_epoch_end=False,
        every_n_train_steps=250,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_top_k=3,
        mode="min",
    )

    final_checkpoint_cb = ModelCheckpoint(
        dirpath=f"./checkpoints/TeTRA-pretrain/",
        filename="final-{epoch}-{train_loss:.4f}-{qfactor:.2f}",
        save_weights_only=True,
        save_on_train_epoch_end=True,
        save_last=True,
    )

    learning_rate_cb = LearningRateMonitor(logging_interval="step")

    # Log the full config directly from OmegaConf
    wandb_logger = WandbLogger(
        project="TeTRA-pretrain",
        name=f"Student[{model_module.student.name}]-Teacher[{model_module.teacher.name}]-Aug[{conf.augmentation_level}]",
        config=OmegaConf.to_container(conf, resolve=True),
    )

    trainer = pl.Trainer(
        enable_progress_bar=conf.pbar,
        strategy=DDPStrategy(find_unused_parameters=True) if conf.use_ddp else "Auto", # Use this for multi-GPU training
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=conf.precision,
        max_epochs=conf.max_epochs,
        callbacks=[checkpoint_cb, final_checkpoint_cb, learning_rate_cb],
        accumulate_grad_batches=conf.accumulate_grad_batches,
        logger=wandb_logger,
        limit_train_batches=25,
        val_check_interval=25, # Check validation every 25 batches
        check_val_every_n_epoch=None # Disable epoch-based validation
    )
    return trainer, model_module


if __name__ == "__main__":
    # Set precision to maximise tensor core usage
    torch.set_float32_matmul_precision("medium")

    args = _argparse()
    conf = OmegaConf.load(args.config)
    print(conf)
    # Setup and run training
    trainer, model_module = setup_training(conf)
    trainer.fit(model_module)
