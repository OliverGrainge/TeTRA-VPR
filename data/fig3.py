import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,
    }
)

# Turn off all grid
plt.rcParams["axes.grid"] = False

DATASET = "Tokyo247"
METHODS_TO_INCLUDE = [
    "ResNet50-BoQ",
    "DinoV2-BoQ",
    "DinoV2-Salad",
    "CosPlaces-D2048",
    "EigenPlaces-D2048",
    "TeTRA-BoQ-DD[1]",
    "TeTRA-SALAD-DD[1]",
    "TeTRA-MixVPR-DD[1]",
]

METHOD_NAMES_MAP = {
    "ResNet50-BoQ": "ResNet50-BoQ",
    "DinoV2-BoQ": "DinoV2-BoQ",
    "DinoV2-Salad": "DinoV2-SALAD",
    "CosPlaces-D2048": "CosPlace",
    "EigenPlaces-D2048": "EigenPlaces",
    "TeTRA-BoQ-DD[1]": "TeTRA-BoQ",
    "TeTRA-Salad-DD[1]": "TeTRA-SALAD",
    "TeTRA-SALAD-DD[1]": "TeTRA-SALAD",
    "TeTRA-MixVPR-DD[1]": "TeTRA-MixVPR",
}

# Define datasets to include in accuracy plot
DATASETS_TO_PLOT = ["Tokyo247"]  # Changed to only include Tokyo247

# Read and prepare data for all datasets
dfs_by_dataset = {}
for dataset in DATASETS_TO_PLOT:
    df_dataset = pd.read_csv("results.csv")
    df_dataset = df_dataset[df_dataset["Dataset"] == dataset].copy()
    df_dataset = df_dataset[df_dataset["Method"].isin(METHODS_TO_INCLUDE)]
    df_dataset["Method"] = df_dataset["Method"].map(METHOD_NAMES_MAP)
    dfs_by_dataset[dataset] = df_dataset

# Use Tokyo247 for sorting and other metrics
df = dfs_by_dataset["Tokyo247"]
df_sorted = df.assign(TotalMemory=df["Model Memory (MB)"] + df["DB Memory (MB)"])
df_sorted = df_sorted.sort_values("TotalMemory")
method_order = df_sorted["Method"].tolist()

x = np.arange(len(method_order))

# Create three subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

# Plot accuracy for each dataset with different markers and colors
colors = ["#70AD47"]  # Only one color needed
markers = ["^"]  # Only one marker needed
for dataset, color, marker in zip(DATASETS_TO_PLOT, colors, markers):
    df_curr = dfs_by_dataset[dataset]
    # Sort according to Tokyo247's memory-based ordering
    df_curr_sorted = df_curr.set_index("Method").reindex(method_order).reset_index()
    ax1.plot(
        x,
        df_curr_sorted["Accuracy (R@1)"],
        marker=marker,
        linestyle="-",
        color=color,
        label="R@1",
    )

# Continue with original plots for latency and memory
ax2.plot(
    x,
    df_sorted["Extraction Latency GPU (ms)"] + df_sorted["Matching Latency (ms)"],
    "s--",
    color="#2F5597",
    label="Total Latency",
)
ax3.plot(x, df_sorted["TotalMemory"], "o-", color="#ED7D31", label="Total Memory")

# Only set x-labels on bottom plot
ax3.set_xticks(x)
ax3.set_xticklabels(df_sorted["Method"], rotation=45, ha="right")

# Set y-labels for each subplot (reordered)
ax1.set_ylabel("Accuracy (R@1)")
ax2.set_ylabel("Total Latency (ms)")
ax3.set_ylabel("Total Memory (MB)")

# Set title only on top subplot
ax1.set_title(f"VPR System: Resource vs. Accuracy on {DATASET}")

# Add grid to all subplots
for ax in [ax1, ax2, ax3]:
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="center right")

# Add annotations for TeTRA-BoQ and CosPlaces across different subplots
for method in ["TeTRA-BoQ", "EigenPlaces"]:
    row = df_sorted[df_sorted["Method"] == method].iloc[0]
    pos = df_sorted[df_sorted["Method"] == method].index.get_loc(row.name)

    accuracy = row["Accuracy (R@1)"]
    memory = row["TotalMemory"]
    latency = row["Extraction Latency GPU (ms)"] + row["Matching Latency (ms)"]

    if method == "TeTRA-BoQ":
        # Accuracy annotation (ax1)
        pos = 2
        print("================== TETRA", pos, accuracy)
        ax1.annotate(
            f"Accuracy: {accuracy:.1f}%",
            xy=(pos, accuracy),
            xytext=(pos, accuracy + 3),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

        ax2.annotate(
            f"Latency: {latency:.0f}ms",
            xy=(pos, latency),
            xytext=(pos, latency + 120),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

        # Memory annotation (ax3)
        ax3.annotate(
            f"Memory: {memory:.0f}MB",
            xy=(pos, memory),
            xytext=(pos, memory + 2600),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

    if method == "EigenPlaces":
        pos = 4
        # Latency annotation (ax2)

        ax1.annotate(
            f"Accuracy: {accuracy:.1f}%",
            xy=(pos, accuracy),
            xytext=(pos, accuracy + 7),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

        ax2.annotate( 
            f"Latency: {latency:.0f}ms",
            xy=(pos, latency),
            xytext=(pos, latency + 100),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

        # Memory annotation (ax3)
        ax3.annotate(
            f"Memory: {memory:.0f}MB",
            xy=(pos, memory),
            xytext=(pos, memory + 2000),
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            ),
        )

# Update legend position for accuracy plot
ax1.legend(loc="center right")

plt.tight_layout()
plt.savefig("figures/fig3.jpg", dpi=300)
plt.show()
