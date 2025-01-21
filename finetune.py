import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import DataConfig, ModelConfig, TeTRAConfig
from dataloaders.TeTRA import TeTRA
from models.helper import get_model


def _get_weights_path(backbone_arch, image_size):
    folders = os.listdir("checkpoints/TeTRA-pretrain")
    for folder in folders:
        if (
            backbone_arch.lower() + str(image_size[0]) in folder.lower()
            and "progressivequant" in folder.lower()
        ):
            weights_folder = os.path.join("checkpoints/TeTRA-pretrain", folder)
            weights_avail = os.listdir(weights_folder)
            if len(weights_avail) > 0:
                if len(weights_avail) > 1:
                    print(
                        f"Multiple weights available for {backbone_arch} {image_size}. Using latest."
                    )
                    return os.path.join(weights_folder, weights_avail[0])
                elif len(weights_avail) == 1:
                    return os.path.join(weights_folder, weights_avail[0])
                else:
                    print(f"No weights available for {backbone_arch} {image_size}")
            else:
                print(f"No weights available for {backbone_arch} {image_size}")
    return None


def load_model(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        desc_divider_factor=args.desc_divider_factor,
    )

    weights_path = _get_weights_path(args.backbone_arch, args.image_size)

    model.backbone.eval()

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise ValueError(f"Checkpoint {args.weights_path} does not exist")

        sd = torch.load(weights_path, weights_only=False)["state_dict"]
        for k, v in sd.items():
            print(k, v.shape)
        backbone_sd = {
            k.replace("backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_sd, strict=False)
        for param in model.backbone.parameters():
            param.requires_grad = False
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
        scheduler_type=args.quant_schedule,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=f"MSLS_val_q_R1",  # msls_val_q_R1
        dirpath=f"./checkpoints/TeTRA-finetune/{str(model)}-DescDividerFactor[{args.desc_divider_factor}]",
        filename="{epoch}-{MSLS_val_q_R1:.2f}",  # msls_val_q_R1
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
        enable_progress_bar=args.pbar,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        check_val_every_n_epoch=args.max_epochs,
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
