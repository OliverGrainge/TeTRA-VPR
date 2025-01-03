from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class DataConfig:
    # dataset directories
    #val_dataset_dir: str = "/home/oliver/datasets_drive/vpr_datasets"
    #train_dataset_dir: str = "/home/oliver/datasets_drive/vpr_datasets/gsv-cities"
    val_dataset_dir: str = "/scratch/oeg1n18/datasets/vpr"
    train_dataset_dir: str = "/scratch/oeg1n18/datasets/vpr/gsvcities"

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Data")
        group.add_argument(
            "--val_dataset_dir", type=str, default=DataConfig.val_dataset_dir
        )
        group.add_argument(
            "--train_dataset_dir", type=str, default=DataConfig.train_dataset_dir
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class ModelConfig:
    # training model
    backbone_arch: str = "ResNet50"
    agg_arch: str = "MixVPR"
    out_dim: int = 2048
    weights_path: Union[str, None] = None

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Model")
        group.add_argument(
            "--backbone_arch", type=str, default=ModelConfig.backbone_arch
        )
        group.add_argument("--agg_arch", type=str, default=ModelConfig.agg_arch)
        group.add_argument("--out_dim", type=int, default=ModelConfig.out_dim)
        group.add_argument("--weights_path", type=str, default=ModelConfig.weights_path)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class EvalConfig:
    # evaluation model
    preset: str = None
    # evaluation dataset
    val_set_names: Tuple[str] = ("sped", "essex")

    # evaluation runtime
    batch_size: int = 32
    num_workers: int = 4
    image_size: Tuple[int] = (224, 224)
    silent: bool = False
    checkpoints_dir: str = "./checkpoints/TeTRA/"
    checkpoint: str = ""

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Eval")
        group.add_argument("--preset", type=str, default=EvalConfig.preset)
        group.add_argument(
            "--val_set_names", type=str, nargs="+", default=EvalConfig.val_set_names
        )
        group.add_argument("--batch_size", type=int, default=EvalConfig.batch_size)
        group.add_argument("--num_workers", type=int, default=EvalConfig.num_workers)
        group.add_argument(
            "--image_size", type=int, nargs=2, default=EvalConfig.image_size
        )
        group.add_argument("--silent", type=bool, default=EvalConfig.silent)
        group.add_argument("--checkpoints_dir", type=str, default=EvalConfig.checkpoints_dir)
        group.add_argument("--checkpoint", type=str, default=EvalConfig.checkpoint)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class DistillConfig:
    # Teacher model settings
    teacher_preset: str = "DinoSalad"

    # Training hyperparameters
    lr: float = 0.0007
    batch_size: int = 128
    accumulate_grad_batches: int = 2
    max_epochs: int = 3

    # Loss and regularization
    mse_loss_mult: float = 1000
    weight_decay_init: float = 0.00
    weight_decay_schedule: str = "constant"
    use_attention: bool = False

    # Data processing
    image_size: Tuple[int] = (224, 224)
    augmentation_level: str = "Moderate"

    # Runtime settings
    num_workers: int = 0
    pbar: bool = False
    checkpoint_dir: str = ""
    val_set_names: Tuple[str] = ("msls","Pitts30k",)
    precision: str = "bf16-mixed"

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Distillation")
        group.add_argument(
            "--teacher_preset", type=str, default=DistillConfig.teacher_preset
        )
        group.add_argument("--lr", type=float, default=DistillConfig.lr)
        group.add_argument("--batch_size", type=int, default=DistillConfig.batch_size)
        group.add_argument(
            "--accumulate_grad_batches",
            type=int,
            default=DistillConfig.accumulate_grad_batches,
        )
        group.add_argument("--max_epochs", type=int, default=DistillConfig.max_epochs)
        group.add_argument(
            "--mse_loss_mult", type=float, default=DistillConfig.mse_loss_mult
        )
        group.add_argument(
            "--weight_decay_init", type=float, default=DistillConfig.weight_decay_init
        )
        group.add_argument(
            "--weight_decay_schedule",
            type=str,
            default=DistillConfig.weight_decay_schedule,
        )
        group.add_argument(
            "--use_attention", type=bool, default=DistillConfig.use_attention
        )
        group.add_argument(
            "--image_size", type=int, nargs=2, default=DistillConfig.image_size
        )
        group.add_argument(
            "--augmentation_level", type=str, default=DistillConfig.augmentation_level
        )
        group.add_argument("--num_workers", type=int, default=DistillConfig.num_workers)
        group.add_argument("--pbar", type=bool, default=DistillConfig.pbar)
        group.add_argument(
            "--checkpoint_dir", type=str, default=DistillConfig.checkpoint_dir
        )
        group.add_argument(
            "--val_set_names", type=str, nargs="+", default=DistillConfig.val_set_names
        )
        group.add_argument("--precision", type=str, default=DistillConfig.precision)
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


@dataclass
class TeTRAConfig:
    # Training hyperparameters
    lr: float = 0.0001
    batch_size: int = 100
    max_epochs: int = 12
    precision: str = "bf16-mixed"

    # Loss and mining settings
    quant_schedule: str = "sigmoid"

    # Data processing
    image_size: Tuple[int] = (224, 224)
    augment_level: str = "LightAugment"

    # Runtime settings
    pbar: bool = False
    num_workers: int = 0
    checkpoint_dir: str = ""

    # backbone freezing
    freeze_all_except_last_n: int = 1

    # Cities
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

    # validation set
    val_set_names: Tuple[str] = ("msls", "pitts30k_val",)

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
        group.add_argument(
            "--checkpoint_dir", type=str, default=TeTRAConfig.checkpoint_dir
        )
        group.add_argument("--cities", type=str, nargs="+", default=TeTRAConfig.cities)
        group.add_argument(
            "--val_set_names", type=str, nargs="+", default=TeTRAConfig.val_set_names
        )
        group.add_argument(
            "--quant_schedule", type=str, default=TeTRAConfig.quant_schedule
        )
        group.add_argument(
            "--freeze_all_except_last_n",
            type=str,
            default=TeTRAConfig.freeze_all_except_last_n,
        )
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            **{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataConfig.add_argparse_args(parser)
    parser = ModelConfig.add_argparse_args(parser)

    # Create separate parsers for Distill and TeTRA configs
    distill_parser = ArgumentParser()
    distill_parser = DataConfig.add_argparse_args(distill_parser)
    distill_parser = ModelConfig.add_argparse_args(distill_parser)
    distill_parser = DistillConfig.add_argparse_args(distill_parser)
    args = distill_parser.parse_args()
    distill_config = DistillConfig.from_argparse_args(args)

    print("")
    print("========= Distill CONFIG =========")
    for key, value in distill_config.__dict__.items():
        print(f"{key}: {value}")

    tetra_parser = ArgumentParser()
    tetra_parser = DataConfig.add_argparse_args(tetra_parser)
    tetra_parser = ModelConfig.add_argparse_args(tetra_parser)
    tetra_parser = TeTRAConfig.add_argparse_args(tetra_parser)
    args = tetra_parser.parse_args()
    tetra_config = TeTRAConfig.from_argparse_args(args)

    print("")
    print("========= TETRA CONFIG =========")
    for key, value in tetra_config.__dict__.items():
        print(f"{key}: {value}")
