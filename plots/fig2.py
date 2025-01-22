import matplotlib.pyplot as plt
import pandas as pd

DATASET = "Tokyo247"

df = pd.read_csv("../data/results.csv")

# Clean up model IDs by removing "DescDividerFactor[1]"
cleaned_ids = df["id"].str.replace("-DescDividerFactor[1]", "")

# Create figure
plt.figure(figsize=(12, 6))

# Create stacked bar chart using cleaned IDs
bar_width = 0.8
bars1 = plt.bar(
    cleaned_ids, df["model_memory_mb"], bar_width, label="Model Memory", color="skyblue"
)
bars2 = plt.bar(
    cleaned_ids,
    df[f"{DATASET}_descriptor_memory_mb"],
    bar_width,
    bottom=df["model_memory_mb"],
    label="Descriptor Memory",
    color="lightcoral",
)

# Add R@1 scores on top of bars
total_heights = df["model_memory_mb"] + df[f"{DATASET}_descriptor_memory_mb"]
for i, (height, score) in enumerate(zip(total_heights, df[f"{DATASET}_R@1"])):
    plt.text(i, height, f"R@1: {score:.1f}%", ha="center", va="bottom", fontsize=9)

# Customize the plot
plt.xlabel("Models")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Usage Breakdown by Model")
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Add grid for better readability
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()
