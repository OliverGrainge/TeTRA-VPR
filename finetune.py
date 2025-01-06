import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import DataConfig, ModelConfig, TeTRAConfig
from dataloaders.TeTRA import TeTRA
from models.helper import get_model


def load_model(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
    )

    if args.weights_path is not None:
        if not os.path.exists(args.weights_path):
            raise ValueError(f"Checkpoint {args.weights_path} does not exist")

        sd = torch.load(args.weights_path, weights_only=False)["state_dict"]
        for k, v in sd.items():
            print(k, v.shape)
        backbone_sd = {
            k.replace("backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }

        model.backbone.load_state_dict(backbone_sd, strict=True)
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model


def setup_training(args, model):
    model_module = TeTRA(
        model,
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        cities=args.cities,
        lr=args.lr,
        scheduler_type=args.quant_schedule,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"Pittsburgh30k_val_q_R1",
        dirpath=f"./checkpoints/TeTRA-finetune/{str(model)}",
        filename="epoch-{epoch}-pitts30k_q_R1{pitts30k_q_R1:.2f}",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="TeTRA-finetune",
        name=f"{str(model)}",
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
        limit_train_batches=10,
    )

    return trainer, model_module


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, TeTRAConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    model = load_model(args)
    trainer, model_module = setup_training(args, model)
    trainer.fit(model_module)
