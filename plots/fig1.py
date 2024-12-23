import os
import matplotlib.pyplot as plt
import pandas as pd 
import sys 
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config import DataConfig


DATASET = "tokyo"

parser = argparse.ArgumentParser()
parser = DataConfig.add_argparse_args(parser)
args = parser.parse_args()

df = pd.read_csv("data/results.csv")
df = df.set_index("id")
print(df.columns)
df = df[(~df.index.str.contains('vit')) | (df['image_size'] == 322)]

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


# Calculate the x-axis values based on the presence of 'preset'
x_values = []
for idx, row in df.iterrows():
    if 'vit' not in idx:
        x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_floating'] / (1024**2)))
    else:
        x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_binary'] / (1024**2)))



# Extract the relevant R@1 values based on the presence of 'preset'
r_at_1_values = []
for idx, row in df.iterrows():
    if 'preset' in row:
        r_at_1_values.append(row[f'{dataset_name}_cosine_R@1'])
    else:
        r_at_1_values.append(row[f'{dataset_name}_hamming_R@1'])

# Plotting
plt.figure(figsize=(10, 6))

# Separate the data based on the presence of 'preset'
preset_x_values = []
preset_r_at_1_values = []
non_preset_x_values = []
non_preset_r_at_1_values = []

for idx, row in df.iterrows():
    if 'vit' not in idx:
        preset_x_values.append(row['model_memory'] + ds_len * (row['descriptor_size_floating'] / (1024**2)))
        preset_r_at_1_values.append(row[f'{dataset_name}_cosine_R@1'])
    else:
        non_preset_x_values.append(row['model_memory'])
        non_preset_r_at_1_values.append(row[f'{dataset_name}_hamming_R@1'])

# Plot preset points
plt.scatter(preset_x_values, preset_r_at_1_values, color='skyblue', label='Preset', marker='o')

# Plot non-preset points
plt.scatter(non_preset_x_values, non_preset_r_at_1_values, color='orange', label='Non-Preset', marker='x')

# Add labels for each point
for i, txt in enumerate(df.index):
    if 'vit' not in txt:
        #continue
        plt.annotate(txt, (x_values[i], r_at_1_values[i]), fontsize=8, alpha=0.7, color='blue')
    #else:
        #plt.annotate(txt, (x_values[i], r_at_1_values[i]), fontsize=8, alpha=0.7, color='red')

plt.xlabel('Sum of Model Memory and Descriptor Size Floating')
plt.ylabel('R@1')
plt.title('R@1 vs Sum of Model Memory and Descriptor Size Floating')
plt.legend()
plt.tight_layout()
plt.show()



