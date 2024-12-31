import os
import matplotlib.pyplot as plt
import pandas as pd 
import sys 
import argparse

# If you want some of the aesthetic benefits of Seaborn:
import seaborn as sns

# This is only necessary if you're executing from a script in a subdirectory
# and want to import local modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DataConfig

DATASET = "tokyo"

parser = argparse.ArgumentParser()
parser = DataConfig.add_argparse_args(parser)
args = parser.parse_args()

df = pd.read_csv("data/results.csv")
df = df.set_index("id")
df = df[(~df.index.str.contains('vit')) | (df['image_size'] == 322)]
df = df[(~df.index.str.contains('gem'))]

if "pitts" in DATASET.lower():
    from dataloaders.val.PittsburghDataset import PittsburghDataset30k
    dataset_name = "Pittsburgh30k"
    ds = PittsburghDataset30k(val_dataset_dir=args.val_dataset_dir, which_set="test")
    ds_len = len(ds)
elif "tokyo" in DATASET.lower():
    dataset_name = "Tokyo247"
    from dataloaders.val.Tokyo247Dataset import Tokyo247
    ds = Tokyo247(val_dataset_dir=args.val_dataset_dir, which_set="test")
    ds_len = len(ds)
elif "msls" in DATASET.lower():
    dataset_name = "MSLS"
    from dataloaders.val.MapillaryDataset import MSLS
    ds = MSLS(val_dataset_dir=args.val_dataset_dir, which_set="test")
    ds_len = len(ds)

# Calculate the x-axis values based on the presence of 'vit' (binary vs floating descriptor sizes).
x_values = []
for idx, row in df.iterrows():
    if 'vit' not in idx:
        x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_floating'] / (1024**2)))
    else:
        x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_binary']   / (1024**2)))

# Calculate R@1 values based on 'preset' (cosine vs hamming).
r_at_1_values = []
for idx, row in df.iterrows():
    if 'preset' in row:
        r_at_1_values.append(row[f'{dataset_name}_cosine_R@1'])
    else:
        r_at_1_values.append(row[f'{dataset_name}_hamming_R@1'])

# Separate the data for plotting
preset_x_values = []
preset_r_at_1_values = []
non_preset_x_values = []
non_preset_r_at_1_values = []

for idx, row in df.iterrows():
    if 'vit' not in idx:
        # Baseline / Preset
        preset_x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_floating'] / (1024**2)))
        preset_r_at_1_values.append(row[f'{dataset_name}_cosine_R@1'])
    else:
        # Tetra / Non-preset
        non_preset_x_values.append(row['model_memory'])
        non_preset_r_at_1_values.append(row[f'{dataset_name}_hamming_R@1'])

# --- Plotting Section ---
# Use a Seaborn style for nicer aesthetics
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))

# Plot Baselines
plt.scatter(
    preset_x_values,
    preset_r_at_1_values,
    color='skyblue',
    edgecolor='black',
    alpha=0.8,
    s=70,
    label='Baselines',
    marker='o'
)

# Plot Tetra
plt.scatter(
    non_preset_x_values,
    non_preset_r_at_1_values,
    color='orange',
    edgecolor='black',
    alpha=0.8,
    s=70,
    label='TeTRA',
    marker='X'
)

# Add labels for each point
for i, txt in enumerate(df.index):
    # Slight offset for annotations so labels don't sit exactly on the points
    x_offset = 0.5
    y_offset = 0.001
    if 'vit' not in txt:
        plt.annotate(
            txt,
            (x_values[i], r_at_1_values[i]),
            xytext=(x_values[i] + x_offset, r_at_1_values[i] + y_offset),
            fontsize=9, 
            alpha=0.8, 
            color='blue' if 'vit' not in txt else 'red',
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        )

plt.xlabel('Memory Consumption (MB)', fontsize=12)
plt.ylabel('R@1', fontsize=12)
plt.title('R@1 vs. VPR Memory Consumption', fontsize=14)
plt.legend(fontsize=11)

# Make layout tight and show grid properly
plt.tight_layout()
plt.show()
