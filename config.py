from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class ModelConfig:
    backbone_arch: str = "ternaryvitbase"  # "ternaryvitbase" or "ternaryvitsmall"
    agg_arch: str = "boq"  # "boq" or "salad", "mixvpr", "gem"
    noramlize: bool = True  # normalize the output descriptors in the forward pass

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Model")
        group.add_argument(
            "--backbone_arch", type=str, default=ModelConfig.backbone_arch
        )
        group.add_argument("--agg_arch", type=str, default=ModelConfig.agg_arch)
        group.add_argument("--noramlize", type=bool, default=ModelConfig.noramlize)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class DistillConfig:
    # add a list of folders containing images to be used for distillation
    # We used the SF_XL panoramas.
    train_dataset_dir: Tuple[str] = ("/path/to/sf_xl/raw/panoramas/",)

    # Training hyperparameters
    lr: float = 0.0004  # pretraining learning rate
    batch_size: int = 128  # training batch size
    accumulate_grad_batches: int = 2  # number of gradient accumulation steps
    max_epochs: int = 30  # number of epochs
    weight_decay: float = 0.01  # weight decay
    use_attn_loss: bool = True  # use attention loss in the distillation loss

    # Data processing
    image_size: Tuple[int] = (322, 322)
    augmentation_level: str = "Severe"

    # Runtime settings
    num_workers: int = 12
    pbar: bool = True
    precision: str = "bf16-mixed"

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Distillation")
        group.add_argument("--lr", type=float, default=DistillConfig.lr)
        group.add_argument("--batch_size", type=int, default=DistillConfig.batch_size)
        group.add_argument(
            "--accumulate_grad_batches",
            type=int,
            default=DistillConfig.accumulate_grad_batches,
        )
        group.add_argument("--max_epochs", type=int, default=DistillConfig.max_epochs)
        group.add_argument(
            "--weight_decay", type=float, default=DistillConfig.weight_decay
        )
        group.add_argument(
            "--use_attn_loss", type=bool, default=DistillConfig.use_attn_loss
        )
        group.add_argument(
            "--image_size", type=int, nargs=2, default=DistillConfig.image_size
        )
        group.add_argument(
            "--augmentation_level", type=str, default=DistillConfig.augmentation_level
        )
        group.add_argument("--num_workers", type=int, default=DistillConfig.num_workers)
        group.add_argument("--pbar", type=bool, default=DistillConfig.pbar)
        group.add_argument("--precision", type=str, default=DistillConfig.precision)
        group.add_argument(
            "--train_dataset_dir", type=str, default=DistillConfig.train_dataset_dir
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class TeTRAConfig:
    # path to GSV-Cities Dataset
    train_dataset_dir: str = "/path/to/gsv-cities/"
    # path to folder with validation sets e.g. msls
    val_dataset_dir: str = "/path/to/vpr_datasets/"

    # Training hyperparameters
    lr: float = 0.0001  # finetuning learning rate
    batch_size: int = 200  # finetuning batch size
    max_epochs: int = 40  # number of epochs
    precision: str = "bf16-mixed"  # precision

    # Loss and mining settings
    quant_schedule: str = (
        "logistic"  # quantization schedule (can choose from "logistic", "linear", "cosine", "none")
    )
    pretrain_checkpoint: str = None  # path to the pretrained checkpoint
    freeze_backbone: bool = False  # freeze the backbone
    freeze_all_except_last_n: int = (
        1  # number of layers to leave unfrozen (if freezing is desired)
    )

    # Data processing
    image_size: Tuple[int] = (322, 322)  # image size
    augment_level: str = (
        "LightAugment"  # augmentation level (can choose from "LightAugment", "SevereAugment")
    )

    # Runtime settings
    pbar: bool = False  # show progress bar during finetuning
    num_workers: int = 12  # number of dataloader workers

    # validation set
    val_set_names: Tuple[str] = ("MSLS",)

    # Cities of GSV-Cities Dataset for finetuning
    cities: Tuple[str] = (
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

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("TeTRA")
        group.add_argument("--lr", type=float, default=TeTRAConfig.lr)
        group.add_argument("--batch_size", type=int, default=TeTRAConfig.batch_size)
        group.add_argument("--max_epochs", type=int, default=TeTRAConfig.max_epochs)
        group.add_argument("--precision", type=str, default=TeTRAConfig.precision)
        group.add_argument(
            "--image_size", type=int, nargs=2, default=TeTRAConfig.image_size
        )
        group.add_argument(
            "--augment_level", type=str, default=TeTRAConfig.augment_level
        )
        group.add_argument("--num_workers", type=int, default=TeTRAConfig.num_workers)
        group.add_argument("--pbar", type=bool, default=TeTRAConfig.pbar)
        group.add_argument("--cities", type=str, nargs="+", default=TeTRAConfig.cities)
        group.add_argument(
            "--val_set_names", type=str, nargs="+", default=TeTRAConfig.val_set_names
        )
        group.add_argument(
            "--quant_schedule", type=str, default=TeTRAConfig.quant_schedule
        )
        group.add_argument(
            "--pretrain_checkpoint", type=str, default=TeTRAConfig.pretrain_checkpoint
        )
        group.add_argument(
            "--freeze_backbone", type=bool, default=TeTRAConfig.freeze_backbone
        )
        group.add_argument(
            "--freeze_all_except_last_n",
            type=str,
            default=TeTRAConfig.freeze_all_except_last_n,
        )
        group.add_argument(
            "--train_dataset_dir", type=str, default=TeTRAConfig.train_dataset_dir
        )

        group.add_argument(
            "--val_dataset_dir", type=str, default=TeTRAConfig.val_dataset_dir
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )
