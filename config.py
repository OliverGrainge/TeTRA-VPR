from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass
class DataConfig:
    # dataset directories
    # val_dataset_dir: str = "/home/oliver/datasets_drive/vpr_datasets" # Desktop
    # val_dataset_dir: str = "/scratch/oeg1n18/datasets/vpr" # HPC
    val_dataset_dir: str = "/Users/olivergrainge/Documents/github/Datasets"  # Laptop

    train_dataset_dir: str = "/scratch/oeg1n18/datasets/vpr/gsvcities"
    # train_dataset_dir: str = (
    #    "/home/oliver/datasets_drive/vpr_datasets/amstertime/images/test/database"
    # )

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
    backbone_arch: str = "Vitsmall"
    agg_arch: str = "salad"
    weights_path: Union[str, None] = None
    desc_divider_factor: int = 1

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Model")
        group.add_argument(
            "--backbone_arch", type=str, default=ModelConfig.backbone_arch
        )
        group.add_argument("--agg_arch", type=str, default=ModelConfig.agg_arch)
        group.add_argument("--weights_path", type=str, default=ModelConfig.weights_path)
        group.add_argument(
            "--desc_divider_factor", type=int, default=ModelConfig.desc_divider_factor
        )
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

    # which evals to run
    accuracy: bool = False
    model_memory: bool = False
    runtime_memory: bool = False
    descriptor_size: bool = False
    feature_extraction_latency: bool = False
    retrieval_latency: bool = False
    dataset_retrieval_latency: bool = False
    compile: bool = False

    # evaluation runtime
    batch_size: int = 32
    num_workers: int = 4
    image_size: Tuple[int] = (224, 224)
    silent: bool = False
    checkpoints_dir: str = "./checkpoints/TeTRA/"

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("Eval")
        group.add_argument("--preset", type=str, default=EvalConfig.preset)
        group.add_argument(
            "--val_set_names", type=str, nargs="+", default=EvalConfig.val_set_names
        )
        group.add_argument(
            "--accuracy", action="store_true", help="Run accuracy evaluation"
        )
        group.add_argument(
            "--model_memory", action="store_true", help="Run model memory evaluation"
        )
        group.add_argument(
            "--runtime_memory",
            action="store_true",
            help="Run runtime memory evaluation",
        )
        group.add_argument(
            "--descriptor_size",
            action="store_true",
            help="Run descriptor size evaluation",
        )
        group.add_argument(
            "--feature_extraction_latency",
            action="store_true",
            help="Run feature extraction latency evaluation",
        )
        group.add_argument(
            "--retrieval_latency",
            action="store_true",
            help="Run retrieval latency evaluation",
        )
        group.add_argument(
            "--dataset_retrieval_latency",
            action="store_true",
            help="Run dataset retrieval latency evaluation",
        )
        group.add_argument("--compile", action="store_true", help="Run compilation")
        group.add_argument("--batch_size", type=int, default=EvalConfig.batch_size)
        group.add_argument("--num_workers", type=int, default=EvalConfig.num_workers)
        group.add_argument(
            "--image_size", type=int, nargs=2, default=EvalConfig.image_size
        )
        group.add_argument("--silent", type=bool, default=EvalConfig.silent)
        group.add_argument(
            "--checkpoints_dir", type=str, default=EvalConfig.checkpoints_dir
        )
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
    weight_decay: float = 0.05
    use_attention: bool = False
    use_progressive_quant: bool = False

    # Data processing
    image_size: Tuple[int] = (224, 224)
    augmentation_level: str = "Severe"

    # Runtime settings
    num_workers: int = 0
    pbar: bool = False
    val_set_names: Tuple[str] = (
        "msls",
        "Pitts30k",
    )
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
            "--weight_decay", type=float, default=DistillConfig.weight_decay
        )
        group.add_argument("--use_attention", action="store_true", help="Use Attention")
        group.add_argument(
            "--use_progressive_quant",
            action="store_true",
            help="Enable progressive quantization",
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
    desc_divider_factor: int = 1
    batch_size: int = 100
    max_epochs: int = 4
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
    val_set_names: Tuple[str] = (
        "msls",
        "Pitts30k",
    )

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        group = parent_parser.add_argument_group("TeTRA")
        group.add_argument("--lr", type=float, default=TeTRAConfig.lr)
        group.add_argument(
            "--desc_divider_factor", type=int, default=TeTRAConfig.desc_divider_factor
        )
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
