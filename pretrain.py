import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler

import utils

torch.set_float32_matmul_precision("medium")


import argparse
import os

import yaml

from dataloaders.EigenPlaces import EigenPlaces
from dataloaders.GSVCities import GSVCities
from models.helper import get_model
from parsers import get_args_parser

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    model = get_model(args.image_size, args.backbone_arch, args.agg_arch, config["Model"], normalize_output=True)
    
    if "gsvcities" == args.training_method.lower():
        model_module = GSVCities(
            config["Training"]["GSVCities"],
            model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            mean_std=IMAGENET_MEAN_STD,
            val_set_names=args.val_set_names,
            search_precision=args.search_precision
        )

    elif "eigenplaces" == args.training_method.lower():
        model_module = EigenPlaces(
            config["Training"]["EigenPlaces"],
            model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            mean_std=IMAGENET_MEAN_STD,
            val_set_names=args.val_set_names,
            search_precision=args.search_precision
        )

    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f"{args.training_method.lower()}/"
        + f"{args.backbone_arch.lower()}"
        + f"_{args.agg_arch.lower()}"
        + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        default_root_dir=f"./Logs/PreTraining/{args.training_method.lower()}/{args.backbone_arch.lower()}_{args.agg_arch.lower()}",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=(
            config["Training"]["EigenPlaces"]["iterations_per_epoch"]
            if args.training_method.lower() == "eigenplaces"
            else None
        ),
    )

    trainer.fit(model_module)
