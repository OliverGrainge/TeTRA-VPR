import argparse
import os

import yaml

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


import argparse


def training_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Training configuration arguments")

        # Model options with config as default
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=config["Training"]["load_checkpoint"],
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--out_dim",
        type=int,
        default=config["Training"]["out_dim"],
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--backbone_arch",
        type=str,
        default=config["Training"]["backbone_arch"],
        help="Backbone architecture for the model",
    )

    # Aggregation architecture
    parser.add_argument(
        "--agg_arch",
        type=str,
        default=config["Training"]["agg_arch"],
        help="Aggregation architecture",
    )

    parser.add_argument(
        "--training_method",
        type=str,
        default=config["Training"]["training_method"],
        help="Type of training gsvcities or eigenplaces",
    )

    # Adding Training configuration arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default=config["Training"]["accelerator"],
        help="Type of accelerator to use (e.g., gpu, cpu)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=config["Training"]["monitor"],
        help="Metric to monitor during training",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=config["Training"]["precision"],
        help="Precision for training (e.g., 16 for mixed precision)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["Training"]["batch_size"],
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=config["Training"]["max_epochs"],
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--search_precision",
        type=str,
        default=config["Training"]["search_precision"],
        help="precision for vector search",
    )

    parser.add_argument(
        "--fast_dev_run",
        type=bool,
        default=config["Training"]["fast_dev_run"],
        help="Run a quick development run",
    )

    parser.add_argument(
        "--val_set_names",
        type=str,
        nargs="+",
        default=["pitts30k_val"],
        help="Validation set names",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config["Training"]["num_workers"],
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=config["Training"]["image_size"],
        help="Size of the images (width, height)",
    )
    parser.add_argument(
        "--freeze_n_blocks",
        type=int,
        default=config["Training"]["freeze_n_blocks"],
        help="number of blocks to freeze weights",
    )

    parser.add_argument(
        "--loss_name",
        type=str,
        default="MultiSimilarityLoss",
        help="choose the loss function name",
    )

    parser.add_argument(
        "--miner_name",
        type=str,
        default="MultiSimilarityMiner",
        help="choose the miner name",
    )

    parser.add_argument(
        "--teacher_preset",
        type=str,
        default=config["Training"]["Distill"]["teacher_preset"],
        help="choose the teacher preset",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Choose a pre-trained benchmark model",
    )

    parser.add_argument(
        "--eval_method",
        type=str,
        default="vpr",
        help="Evaluation method",
    )

    parser.add_argument(
        "--use_attention",
        type=bool,
        default=config["Training"]["Distill"]["use_attention"],
        help="Use attention",
    )

    parser.add_argument(
        "--pbar",
        type=bool,
        default=config["Training"]["pbar"],
        help="Use progress bar",
    )

    parser.add_argument(
        "--distill_lr",
        type=float,
        default=config["Training"]["Distill"]["distill_lr"],
        help="Learning rate for distillation",
    )

    parser.add_argument(
        "--weight_decay_init",
        type=float,
        default=config["Training"]["Distill"]["weight_decay_init"],
        help="Weight decay scale",
    )

    parser.add_argument(
        "--weight_decay_schedule",
        type=str,
        default=config["Training"]["Distill"]["weight_decay_schedule"],
        help="Weight decay schedule",
    )

    parser.add_argument(
        "--mse_loss_mult",
        type=float,
        default=config["Training"]["Distill"]["mse_loss_mult"],
        help="MSE loss scale",
    )

    parser.add_argument(
        "--augment_level",
        type=str,
        default=config["Training"]["Distill"]["augment_level"],
        help="Augmentation level",
    )
    return parser


def get_args_parser():
    parser = argparse.ArgumentParser(description="Model and Training arguments")
    parser = training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()

    # Parse the arguments
    args = parser.parse_args()
    print(args)
