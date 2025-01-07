import os
import argparse
from PIL import Image
from tqdm import tqdm

# Add argument parser
parser = argparse.ArgumentParser(description='Clean dataset by removing corrupted images')
parser.add_argument('--dataset_root', type=str, required=True,
                    help='Path to the dataset directory containing images')

args = parser.parse_args()
dataset_root = args.dataset_root

# Get total number of files first
total_files = sum([len(files) for _, _, files in os.walk(dataset_root)])

# Replace the simple walk loop with a progress bar
with tqdm(total=total_files, desc="Checking images") as pbar:
    for root, _, files in os.walk(dataset_root):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with Image.open(filepath) as img:
                    img.convert("RGB")  # force load
            except OSError:
                print(f"Removing corrupted image: {filepath}")
                os.remove(filepath)
            pbar.update(1)