import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


torch.set_float32_matmul_precision("medium")


import argparse
import os

import yaml

from dataloaders.EigenPlaces import EigenPlaces
from dataloaders.GSVCities import GSVCities
from dataloaders.Distill import VPRDistill
from dataloaders.ImageNet import ImageNet
from models.helper import get_model
from parsers import get_args_parser


config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def freeze_blocks(model, blocks=None):
    """
    Freezes specific blocks of the transformer layers in the model.

    Args:
        model: The model containing transformer blocks.
        block_indices: List of block indices to freeze. If None, freezes all blocks.
    """
    # Freeze specific transformer blocks
    for idx, block in enumerate(model.backbone.transformer.layers):
        if blocks is None or idx < blocks:
            for param in block.parameters():
                param.requires_grad = False
            print(f"Block {idx} frozen.")
        else:
            print(f"Block {idx} left unfrozen.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if (
        "vit" in args.backbone_arch
        or "cct" in args.backbone_arch
        or "dino" in args.backbone_arch
    ):
        MEAN_STD = VIT_MEAN_STD
    else:
        MEAN_STD = IMAGENET_MEAN_STD

    if "gsvcities" == args.training_method.lower():
        model = get_model(
            args.image_size,
            args.backbone_arch,
            args.agg_arch,
            out_dim=args.out_dim,
            normalize_output=False,
        )
        if args.load_checkpoint != "":
            sd = torch.load(args.load_checkpoint)
            sd = sd["state_dict"]
            new_sd = {}
            for key, value in sd.items():
                if key != "fc.weight" and key != "fc.bias":
                    new_sd[key.replace("model.", "")] = value
            model.load_state_dict(new_sd, strict=False)

            freeze_blocks(model, args.freeze_n_blocks)

        model_module = GSVCities(
            config["Training"]["GSVCities"],
            model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            mean_std=MEAN_STD,
            val_set_names=args.val_set_names,
            loss_name=args.loss_name,
            miner_name=args.miner_name,
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="pitts30k_val/binary_R1",
            filename=f"{args.training_method.lower()}/"
            + f"{args.backbone_arch.lower()}"
            + f"_{args.agg_arch.lower()}"
            + f"_{args.loss_name}"
            + f"_{args.miner_name}"
            + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )

    if "distill" in args.training_method.lower():
        model_module = VPRDistill(
            config["Training"]["Distill"],
            args=args,
            student_backbone_arch=args.backbone_arch,
            student_agg_arch=args.agg_arch,
            teacher_preset=args.teacher_preset,
            use_attention=args.use_attention,
            
        )

        mycheckpoint_cb = ModelCheckpoint(
            monitor=f"{args.val_set_names[0]}_R1",
            filename=f"{args.backbone_arch.lower()}"
            + f"_{args.agg_arch.lower()}" + f"_{args.teacher_preset.lower()}"
            + "_epoch({epoch:02d})_step({step:04d})_loss{train_loss:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )

        print("============================= using this checkpoint")

    elif "eigenplaces" == args.training_method.lower():
        model = get_model(
            args.image_size,
            args.backbone_arch,
            args.agg_arch,
            preset=args.preset,
            out_dim=args.out_dim,
            normalize_output=True,
        )

        model_module = EigenPlaces(
            config["Training"]["EigenPlaces"],
            model,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            mean_std=MEAN_STD,
            val_set_names=args.val_set_names,
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="R5",
            filename=f"{args.training_method.lower()}/{args.backbone_arch.lower()}_{args.agg_arch.lower()}/"
            + f"{args.backbone_arch.lower()}"
            + f"_{args.agg_arch.lower()}"
            + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )
    elif "imagenet" in args.training_method.lower():
        model = get_model(
            args.image_size,
            args.backbone_arch,
            args.agg_arch,
            out_dim=1000,
            normalize_output=False,
        )

        if "ternary" in args.backbone_arch.lower():
            opt_type = "bitnet"
            if "base" in args.backbone_arch.lower():
                lr = 5e-4
            else:
                lr = 1e-3
        else:
            opt_type = "float"
            lr = 3e-4

        model_module = ImageNet(
            model=model,
            batch_size=args.batch_size,
            workers=args.num_workers,
            lr=lr,
            max_epochs=args.max_epochs,
            opt_type=opt_type,
            warmup_epochs=5,
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="val_acc5",
            filename=f"{args.backbone_arch.lower()}"
            + f"_{args.agg_arch.lower()}"
            + "_epoch({epoch:02d})_step({step:04d})_A1[{val_acc1:.4f}]_A5[{val_acc5:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")


    trainer = pl.Trainer(
        enable_progress_bar=False,
        strategy="auto",
        devices=1,
        accelerator="auto",
        default_root_dir=f"./Logs/PreTraining/{args.training_method.lower()}/{args.backbone_arch.lower()}_{args.agg_arch.lower()}",
        num_sanity_val_steps=0,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, mycheckpoint_cb],
        fast_dev_run=args.fast_dev_run,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=0.02 if "distill" in args.training_method else 1.0,
    )

    trainer.fit(model_module)