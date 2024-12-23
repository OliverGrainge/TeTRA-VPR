import argparse

import pytorch_lightning as pl
import torch
from tabulate import tabulate

from config import DataConfig, EvalConfig, ModelConfig
from dataloaders.VPREval import VPREval
from models.helper import get_model
from models.transforms import get_transform
import pandas as pd
import os
import logging
import re

RESULTS_FILE = "data/results.csv"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

def save_results(accuracy_results, resource_results):
    """
    Save results to the CSV file. Merge results if an entry with the same `id` already exists,
    and handle new columns dynamically.
    """
    # Combine accuracy and resource results into a single dictionary
    combined_results = {**accuracy_results, **resource_results}

    # Create a DataFrame from the results
    new_df = pd.DataFrame([combined_results])  # Wrap in a list to ensure it's a single row

    if os.path.exists(RESULTS_FILE):
        # Load the existing results file
        existing_df = pd.read_csv(RESULTS_FILE)

        # Ensure all new columns in `new_df` are added to `existing_df`
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None

        # Check if the `id` exists in the current results
        if combined_results["id"] in existing_df["id"].values:
            # Find the row index for the existing entry
            row_index = existing_df[existing_df["id"] == combined_results["id"]].index[0]

            # Merge the new results into the existing row
            for key, value in combined_results.items():
                if key in existing_df.columns:  # Check if column exists
                    if pd.isna(existing_df.at[row_index, key]) or existing_df.at[row_index, key] == "":
                        existing_df.at[row_index, key] = value
        else:
            # Remove empty or all-NA columns from new_df before concatenation
            new_df = new_df.dropna(axis=1, how='all')
            # Append the new results to the existing DataFrame
            existing_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, start with the new results DataFrame
        existing_df = new_df

    # Save back to the CSV file
    existing_df.to_csv(RESULTS_FILE, index=False)


def _get_checkpoint_path(args): 
    if args.checkpoints_dir is not None: 
        model_folders = os.listdir(args.checkpoints_dir)
        selected_folder = None
        for folder in model_folders: 
            if (args.backbone_arch.lower() in folder.lower() and 
                args.agg_arch.lower() in folder.lower() and 
                str(args.image_size[0]) in folder.lower()):  # Convert to string
                selected_folder = folder 
                break 
        if selected_folder is None:
            raise ValueError(f"Checkpoint path for {args.backbone_arch} {args.agg_arch} {args.image_size[0]} not found")
    
    recall_scores = []
    model_dir = os.path.join(args.checkpoints_dir, selected_folder)
    model_paths = os.listdir(model_dir)
    for filename in os.listdir(model_dir):
        # Check if the file matches the expected pattern
        if filename.endswith(".ckpt"):
            # Use a regex to extract the Recall@1 score
            filename = filename.replace(".ckpt", "")
            match = re.search(r"R1=([0-9.]+)", filename)
            if match:
                recall = float(match.group(1))  # Convert to float
                recall_scores.append((filename+".ckpt", recall))

    if not recall_scores:
        raise ValueError("No valid checkpoint files found with recall scores in the specified directory.")

    model_path = sorted(recall_scores, key=lambda x: x[1], reverse=True)[0][0]
    return os.path.join(args.checkpoints_dir, selected_folder, model_path)


def _load_model_and_transform(args):
    if args.preset is not None:
        model = get_model(preset=args.preset)
        transform = get_transform(preset=args.preset)

        return model, transform
    else:
        model = get_model(backbone_arch=args.backbone_arch, agg_arch=args.agg_arch, image_size=args.image_size)
        transform = get_transform(augmentation_level="None", image_size=args.image_size)

    checkpoint_path = _get_checkpoint_path(args)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "freeze"):
        model.freeze()
    return model, transform


def eval(args):
    model, transform = _load_model_and_transform(args)
    module = VPREval(
        model=model,
        transform=transform,
        val_set_names=args.val_set_names,
        val_dataset_dir=args.val_dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        silent=args.silent
    )

    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        enable_progress_bar=not args.silent,
        accelerator="auto",
        precision="bf16-mixed",
        max_epochs=1,  # Set the number of epochs
        logger=False,  # Disable logging if not needed
    )

    # Use the trainer to validate the module
    trainer.validate(module)
    return module.results

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, EvalConfig]:
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()


    if args.silent:
        logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
    results = eval(args)
    accuracy_results = results["accuracy"]
    resource_results = results["resources"]

    if args.preset is not None:
        accuracy_results["id"] = args.preset 
        accuracy_results["preset"] = args.preset 
        accuracy_results["backbone_arch"] = None
        accuracy_results["agg_arch"] = None
        accuracy_results["image_size"] = args.image_size[0]

    else: 
        accuracy_results["id"] = args.backbone_arch + "_" + args.agg_arch + "_" + str(args.image_size[0])
        accuracy_results["preset"] = None
        accuracy_results["backbone_arch"] = args.backbone_arch
        accuracy_results["agg_arch"] = args.agg_arch
        accuracy_results["image_size"] = args.image_size[0]
    print("==============================================================", accuracy_results["id"])
    save_results(accuracy_results, resource_results)
