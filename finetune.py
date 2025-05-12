import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from config import ModelConfig, TeTRAConfig
from dataloaders.TeTRA import TeTRA
from models.helper import get_model

CITIES = (
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",
)


def _parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="myyaml.yaml")
    return parser.parse_args()


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def _freeze_backbone(model: nn.Module, unfreeze_n_last_layers: int = 1):
    backbone = model.backbone

    _freeze_module(backbone)

    # set dropout to 0 for all layers except the last unfreeze_n_last_layers
    for block in backbone.transformer.layers[:-unfreeze_n_last_layers]:
        for name, module in block.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    # only train the last unfreeze_n_last_layers
    for block in backbone.transformer.layers[-unfreeze_n_last_layers:]:
        _unfreeze_module(block)

    # make sure the backbone is in fully quantized mode
    for module in model.modules():
        if hasattr(module, "set_qfactor"):
            module.set_qfactor(1.0)

    return model


def _load_backbone_weights(model: nn.Module, pretrain_checkpoint_path: str):
    assert os.path.exists(
        pretrain_checkpoint_path
    ), f"Backbone weights path {pretrain_checkpoint_path} does not exist"
    sd = torch.load(pretrain_checkpoint_path, weights_only=False)["state_dict"]
    new_sd = {}
    for key, value in sd.items():
        if key.startswith("student"):
            new_sd[key.replace("student.", "")] = value

    model.backbone.load_state_dict(new_sd)
    return model


def load_model(conf):
    model = get_model(
        image_size=conf.image_size,
        backbone_arch=conf.backbone_arch,
        agg_arch=conf.agg_arch,
    )

    model = _load_backbone_weights(
        model, pretrain_checkpoint_path=conf.pretrain_checkpoint
    )
    model = _freeze_backbone(model, unfreeze_n_last_layers=1)
    model.train()
    return model


def setup_training(conf, model):

    model_module = TeTRA(
        model,
        train_dataset_dir=conf.train_dataset_dir,
        val_dataset_dir=conf.val_dataset_dir,
        val_set_names=conf.val_set_names,
        batch_size=conf.batch_size,
        image_size=conf.image_size,
        num_workers=conf.num_workers,
        cities=CITIES,
        lr=conf.lr,
        quant_schedule=conf.quant_schedule,
    )

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    checkpoint_cb = ModelCheckpoint(
        monitor=f"MSLS_binary_R1",
        dirpath=f"./checkpoints/TeTRA-finetune/{config_name}",
        filename="{epoch}-{MSLS_binary_R1:.2f}",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        project="TeTRA-finetune",
        name=model.name,
    )

    trainer = pl.Trainer(
        enable_progress_bar=conf.pbar,
        strategy="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        precision=conf.precision,
        max_epochs=conf.max_epochs,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        limit_train_batches=25,
    )

    return trainer, model_module


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    # Parse arguments
    args = _parseargs()
    conf = OmegaConf.load(args.config)
    model = load_model(conf)
    trainer, model_module = setup_training(conf, model)
    trainer.fit(model_module)
