import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import torchvision.transforms as T

from dataloaders.ImageNet import ImageNet
from models.helper import get_model, get_transform
from parsers import get_args_parser


def measure_latency(model, input_tensor, num_runs=100, warmup_runs=10, verbose=True):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run on a CUDA-enabled device."
        )

    device = torch.device("cuda")
    model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Warm-up runs
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            torch.cuda.synchronize()

        # Timing runs
        latencies = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(input_tensor)
            end_event.record()

            # Wait for the events to be recorded
            torch.cuda.synchronize()

            # Calculate elapsed time in milliseconds
            elapsed_time = start_event.elapsed_time(end_event)
            latencies.append(elapsed_time)

    average_latency = sum(latencies) / len(latencies)
    if verbose:
        print(" ")
        print(" Average Latency: ", average_latency)
        print(" ")
    return average_latency


def measure_memory(model, verbose=True):
    state_dict = model.state_dict()
    total_bytes = 0

    for key, param in state_dict.items():
        # Ensure the parameter is a tensor
        if isinstance(param, torch.Tensor):
            param_size = param.numel() * param.element_size()
            total_bytes += param_size
        else:
            raise TypeError(
                f"Expected torch.Tensor for key '{key}', but got {type(param)}"
            )

    # Convert bytes to megabytes
    total_megabytes = total_bytes / (1024**2)
    if verbose:
        print(" ")
        print(" Model Size: ", total_megabytes)
        print(" ")
    return total_megabytes


def eval_vpr(args):
    from dataloaders.VPREval import VPREval

    if args.preset is not None:
        model = get_model(preset=args.preset)
        transform = get_transform(args.preset)
    else:
        model = get_model(
            args.image_size,
            args.backbone_arch,
            args.agg_arch,
            out_dim=args.out_dim,
            normalize_output=False,
        )

        model.load_state_dict(torch.load(args.load_checkpoint))
        transform = T.Compose(
            [
                T.Resize(args.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    module = VPREval(
        model,
        transform,
        args.val_set_names,
        args.search_precision,
        args.batch_size,
        args.num_workers,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        precision="32",
        devices=1
    )
    results = trainer.validate(module)
    return results


def eval_imagenet(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        out_dim=1000,
        normalize_output=False,
    )
    model.eval()
    module = ImageNet.load_from_checkpoint(
        checkpoint_path=args.load_checkpoint, model=model
    )

    measure_latency(model, torch.randn(1, 3, 224, 224))
    measure_memory(model)
    trainer = pl.Trainer(precision="bf32-true", devices=1)
    trainer.test(module)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if args.eval_method == "vpr":
        eval_vpr(args)
    elif args.eval_method == "imagenet":
        eval_imagenet(args)
    else:
        raise ValueError(f"Invalid evaluation method: {args.eval_method}")
