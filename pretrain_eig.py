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
from models import helper
from parsers import dataloader_arguments, model_arguments, training_arguments

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

if __name__ == "__main__":
    # the datamodule contains train and validation dataloaders,
    # refer to ./dataloader/GSVCitiesDataloader.py for details
    # if you want to train on specific cities, you can comment/uncomment
    # cities from the list TRAIN_CITIES
    parser = argparse.ArgumentParser(
        description="Model, Training, and Dataloader arguments"
    )
    parser = model_arguments(parser)
    parser = training_arguments(parser)
    parser = dataloader_arguments(parser)
    args = parser.parse_args()

    model = torch.nn.Sequential(
        helper.get_backbone(args.backbone_arch, config["Model"]["backbone_config"]),
        helper.get_aggregator(args.agg_arch, config["Model"]["agg_config"]),
    )

    model = EigenPlaces(
        config,
        model,
        dataset_size="small",
        batch_size=args.batch_size,
        output_dim=512,
        shuffle_all=args.shuffle_all,
        image_size=args.image_size,
        num_workers=6,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val"],
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f"{args.backbone_arch}"
        + f"_{args.agg_arch}"
        + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )

    # Instantiate a trainer with parsed arguments
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,  # gpu
        default_root_dir=f"./Logs/PreTraining/{model.backbone_arch}",  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before starting training
        precision="bf16-mixed",  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_cb
        ],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,  # comment if you want to start training the network and saving checkpoints
        limit_train_batches=500,
    )

    # Run the trainer with the model and datamodule
    trainer.fit(model)
