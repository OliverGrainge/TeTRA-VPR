import argparse
import logging
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image

from config import DataConfig, EvalConfig, ModelConfig
from dataloaders.Eval import evaluate
from models.helper import get_model
from models.transforms import get_transform

RESULTS_FILE = "data/results.csv"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)


def save_results(results):
    """
    Save results to the CSV file. Merge results if an entry with the same `id` already exists,
    and handle new columns dynamically.
    """
    # Create a DataFrame from the results
    new_df = pd.DataFrame([results])  # Wrap in a list to ensure it's a single row

    if os.path.exists(RESULTS_FILE):
        # Load the existing results file
        existing_df = pd.read_csv(RESULTS_FILE)

        # Ensure all new columns in `new_df` are added to `existing_df`
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None

        # Check if the `id` exists in the current results
        if results["id"] in existing_df["id"].values:
            # Find the row index for the existing entry
            row_index = existing_df[existing_df["id"] == results["id"]].index[0]

            # Merge the new results into the existing row
            for key, value in results.items():
                if key in existing_df.columns:  # Check if column exists
                    if (
                        pd.isna(existing_df.at[row_index, key])
                        or existing_df.at[row_index, key] == ""
                    ):
                        existing_df.at[row_index, key] = value
        else:
            # Remove empty or all-NA columns from new_df before concatenation
            new_df = new_df.dropna(axis=1, how="all")
            # Append the new results to the existing DataFrame
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, start with the new results DataFrame
        existing_df = new_df

    # Save back to the CSV file
    existing_df.to_csv(RESULTS_FILE, index=False)


def _get_weights_path(backbone_arch, agg_arch, image_size, desc_divider_factor):
    folders = os.listdir("checkpoints/TeTRA-finetune")
    for folder in folders:
        if (
            backbone_arch.lower() + str(image_size[0]) in folder.lower()
            and agg_arch.lower() in folder.lower()
            and f"descdividerfactor[{desc_divider_factor}]" in folder.lower()
        ):
            weights_folder = os.path.join("checkpoints/TeTRA-finetune", folder)
            weights_avail = os.listdir(weights_folder)
            if len(weights_avail) > 0:
                if len(weights_avail) > 1:
                    print(
                        f"Multiple weights available for {backbone_arch} {image_size}. Using latest."
                    )
                    return os.path.join(weights_folder, weights_avail[0])
                elif len(weights_avail) == 1:
                    return os.path.join(weights_folder, weights_avail[0])
                else:
                    print(f"No weights available for {backbone_arch} {image_size}")
            else:
                print(f"No weights available for {backbone_arch} {image_size}")
    return None


def _load_model_and_transform(args):
    if args.preset is not None:
        model = get_model(preset=args.preset)
        transform = get_transform(preset=args.preset)
        return model, transform
    else:
        model = get_model(
            backbone_arch=args.backbone_arch,
            agg_arch=args.agg_arch,
            image_size=args.image_size,
            desc_divider_factor=args.desc_divider_factor,
        )
        transform = get_transform(augmentation_level="None", image_size=args.image_size)

    model.eval()
    checkpoint_path = _get_weights_path(
        args.backbone_arch, args.agg_arch, args.image_size, args.desc_divider_factor
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    return model, transform


def _get_model_id(args):
    if args.preset is not None:
        return args.preset
    else:
        return f"{args.backbone_arch}{args.image_size[0]}_{args.agg_arch}-DescDividerFactor[{args.desc_divider_factor}]"


def _get_example_input(args, transform):
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    example_input = transform(img).unsqueeze(0)
    return example_input.to(args.device)


def _prepare_model(args, model):
    model.to(args.device)
    if args.compile:
        if hasattr(model, "deploy"):
            model.deploy()
    else:
        model.eval()
    return model


def _detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def eval(args):
    args.device = _detect_device()
    model, transform = _load_model_and_transform(args)
    example_input = _get_example_input(args, transform)
    model = _prepare_model(args, model)
    results = evaluate(args, model, example_input)
    results["id"] = _get_model_id(args)
    save_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, EvalConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    eval(args)
