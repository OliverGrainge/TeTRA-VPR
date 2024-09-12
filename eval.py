import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

from dataloaders.GSVCities import GSVCitiesDataModule
from parsers import (
    dataloader_arguments,
    model_arguments,
    quantize_arguments,
    training_arguments,
)
from pretrain_gsv import VPRModel

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "NeuroCompress"))
)


from NeuroPress import QLayers as Q
from NeuroPress import freeze, postquantize


def get_qlayers(args):
    qlinear = getattr(Q, args.qlinear) if args.qlinear else None
    qconv = getattr(Q, args.qconv) if args.qconv else None
    return qlinear, qconv


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Model, Quantize arguments")
    parser = model_arguments(parser)
    parser = quantize_arguments(parser)
    parser = dataloader_arguments(parser)
    parser = training_arguments(parser)
    args = parser.parse_args()
    args.load_checkpoint = "/home/oliver/Documents/github/QuantPlaceFinder/Logs/PreTraining/vit/lightning_logs/version_0/checkpoints/vit_cls_MultiSimilarityLoss_epoch(15)_step(4688)_R1[0.8061]_R5[0.9435].ckpt"

    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        cities=args.cities,
        shuffle_all=args.shuffle_all,
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=args.image_size,
        num_workers=args.num_workers,
        show_data_stats=False,
        val_set_names=args.val_set_names,
    )

    assert os.path.exists(args.load_checkpoint)
    model = VPRModel.load_from_checkpoint(args.load_checkpoint)
    model.search_precision = args.search_precision
    print(args.search_precision)
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(
        "==================================================================================="
    )
    print(
        "================================== Full Precision ======================================"
    )
    print(
        "==================================================================================="
    )
    trainer = pl.Trainer()
    # metrics = trainer.validate(model=model, datamodule=datamodule, verbose=True)

    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(
        "==================================================================================="
    )
    print(
        "================================== Quantized ======================================"
    )
    print(
        "==================================================================================="
    )
    postquantize(model.backbone, qlinear=Q.LinearWTA16)
    freeze(model.backbone)
    trainer = pl.Trainer()
    qmetrics = trainer.validate(model=model, datamodule=datamodule, verbose=True)
    # print(model)
