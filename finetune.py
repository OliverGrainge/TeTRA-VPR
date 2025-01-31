import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
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
        normalize_output=True,
    )

    if args.weights_path is not None:
        if not os.path.exists(args.weights_path):
            raise ValueError(f"Checkpoint {args.weights_path} does not exist")

        sd = torch.load(args.weights_path)["state_dict"]
        backbone_sd = {
            k.replace("backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }

        # load_state_dict returns a NamedTuple with missing_keys and unexpected_keys
        model.backbone.load_state_dict(backbone_sd, strict=True)
        # for param in model.backbone.parameters():
        #    param.requires_grad = False
        # model.backbone.freeze()
        model.backbone.freeze_all_except_last_n(1)
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
        monitor=f"MSLS_val_q_R1",
        dirpath=_get_checkpoint_dir(args),
        filename="epoch-{epoch}-msls_q_R1{pitts30k_q_R1:.2f}",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="TeTRA",
        name=_get_wandb_run_name(args),
    )

    trainer = pl.Trainer(
        enable_progress_bar=False,
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
    return f"./checkpoints/TeTRA/backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_aug[{args.augment_level.lower()}]_quant_schedule[{args.quant_schedule}]_res[{args.image_size[0]}x{args.image_size[1]}]"


def _get_wandb_run_name(args):
    return f"backbone[{args.backbone_arch.lower()}]_agg[{args.agg_arch.lower()}]_aug[{args.augment_level.lower()}]_quant_schedule[{args.quant_schedule}]_res[{args.image_size[0]}x{args.image_size[1]}]"


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, TeTRAConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    model = load_model(args)
    trainer, model_module = setup_training(args, model)
    trainer.fit(model_module)
