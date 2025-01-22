import matplotlib.pyplot as plt
import pandas as pd

DATASET = "Pittsburgh30k"

df = pd.read_csv("../data/results.csv")
print(df.columns)

# Separate data points based on TeTRA, excluding specific models
base_mask = df["id"].str.contains("baset224|smallt224", case=False)
tetra_mask = df["id"].str.contains(
    "TeTRA-", case=False
)  # Added $ to match exact ending
non_tetra = df[~tetra_mask]
tetra = df[tetra_mask & ~base_mask]


# Create the scatter plot with different markers
plt.figure(figsize=(10, 6))
plt.scatter(
    non_tetra[f"{DATASET}_total_memory_mb"],
    non_tetra[f"{DATASET}_R@1"],
    marker="o",
    alpha=0.6,
    label="Other Models",
)
plt.scatter(
    tetra[f"{DATASET}_total_memory_mb"],
    tetra[f"{DATASET}_R@1"],
    marker="^",
    alpha=0.6,
    label="TeTRA Models",
)

# Add labels and title
plt.xlabel("Total Memory (MB)", fontsize=12)
plt.ylabel(f"{DATASET} R@1", fontsize=12)
plt.title(f"{DATASET} R@1 vs Total Memory Usage", fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Optional: Annotate points with model names
for i, txt in enumerate(df["id"]):
    # Check if the ID contains 'vitbaset' or 'vitsmallt' (case insensitive)
    color = (
        "blue" if ("vitbaset" in txt.lower() or "vitsmallt" in txt.lower()) else "red"
    )
    pos = (
        (5, 5) if ("vitbaset" in txt.lower() or "vitsmallt" in txt.lower()) else (5, -5)
    )
    #
    # plt.annotate(txt, (df[f'{DATASET}_total_memory_mb'][i], df[f'{DATASET}_R@1'][i]),
    #            xytext=pos, textcoords='offset points', fontsize=6, color=color)

# Add legend
plt.legend()

plt.tight_layout()
plt.show()
