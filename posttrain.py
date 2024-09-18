import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pretrain_gsv import VPRModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler

import utils
from dataloaders.GSVCities import GSVCitiesDataModule
from models import helper
from parsers import (
    dataloader_arguments,
    model_arguments,
    quantize_arguments,
    training_arguments,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "NeuroCompress"))
)
import yaml
from NeuroPress import QLayers as Q
from NeuroPress import freeze, postquantize

torch.set_float32_matmul_precision("medium")

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


def get_qlayers(args):
    qlinear = getattr(Q, args.qlinear) if args.qlinear else None
    return qlinear


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
    parser = quantize_arguments(parser)
    args = parser.parse_args()

    # Instantiate the datamodule with parsed arguments
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        cities=args.cities,
        shuffle_all=args.shuffle_all,
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=args.image_size,
        num_workers=args.num_workers,
        show_data_stats=args.show_data_stats,
        val_set_names=args.val_set_names,
    )

    model = VPRModel.load_from_checkpoint(args.load_checkpoint)

    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on pittsburgh val
    checkpoint_cb = ModelCheckpoint(
        monitor=args.monitor,
        filename=f"{model.backbone_arch}"
        + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode="max",
    )

    # Instantiate a trainer with parsed arguments
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,  # gpu
        default_root_dir=f"./Logs/PostTraining/{model.backbone_arch}",  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before starting training
        precision=args.precision,  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[
            checkpoint_cb
        ],  # we run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,  # comment if you want to start training the network and saving checkpoints
    )
    # trainer.validate(model=model, datamodule=datamodule)
    # Run the trainer with the model and datamodule

    qlinear = get_qlayers(args)
    print(f"Quantizing with {qlinear} weights")

    # qlayer_map = {}
    # for name, layer in model.named_modules():
    # print(name)
    # if "fc1" in name:
    # if "11" in name or "10" in name or "9" in name or "8" in name:
    # qlayer_map[layer] = qlinear
    # print("================================================================================= Quantizing")
    # if "fc2" in name:
    # if "11" in name or "10" in name or "9" in name or "8" in name:
    # qlayer_map[layer] = qlinear
    # old_weights = layer.weight.data.detach().cpu().numpy().flatten()
    # print("================================================================================= Quantizing")

    postquantize(model.backbone, qlinear=qlinear)
    freeze(model.backbone)

    trainer.validate(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
