import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a consistent style and font settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,
    }
)
plt.rcParams["axes.grid"] = False

# Define methods and mapping
METHODS_TO_INCLUDE = [
    "DinoV2-BoQ",
    "DinoV2-Salad",
    "CosPlaces-D2048",
    "TeTRA-BoQ-DD[1]",
]

METHOD_NAMES_MAP = {
    "ResNet50-BoQ": "ResNet50-BoQ",
    "DinoV2-BoQ": "DinoV2-BoQ",
    "DinoV2-Salad": "DinoV2-Salad",
    "CosPlaces-D2048": "CosPlaces",
    "EigenPlaces-D2048": "EigenPlaces",
    "TeTRA-BoQ-DD[1]": "TeTRA-BoQ",
    "TeTRA-BoQ-DD[2]": "TeTRA-BoQ",
}

# Define the original metrics and how we want to rename them for the plot
metrics_original = ["Accuracy (R@1)", "Total Memory (MB)", "Total Latency (ms)"]
metric_rename_map = {
    "Accuracy (R@1)": "R@1",
    "Total Memory (MB)": "Mem",
    "Total Latency (ms)": "Lat",
}
metrics_final = [metric_rename_map[m] for m in metrics_original]

# Define whether higher is better ("max") or lower is better ("min")
metric_directions = {
    "Accuracy (R@1)": "max",  # higher is better
    "Total Memory (MB)": "min",  # less memory is better
    "Total Latency (ms)": "min",  # less latency is better
}

# Load the full results CSV (assumes results.csv contains all datasets)
df_all = pd.read_csv("results.csv")

# Update dataset list: Remove Essex3in1.
datasets = [
    "Pitts30k",
    "MSLS",
    "Tokyo247",  # These three will be in the top row.
    "SVOX-night",
    "SVOX-rain",
    "SVOX-snow",
    "SVOX-sun",  # These four will be in the bottom row.
]

# Create a figure with increased height
fig = plt.figure(figsize=(28, 8))
gs = gridspec.GridSpec(
    3, 20, figure=fig, hspace=-0.02, wspace=-0.025, top=0.92
)  # Changed hspace from 0.0 to -0.4

# Define gridspec slices for each dataset's subplot.
# Top row (row 0): 3 subplots, each 4 columns wide, centered (columns 4–8, 8–12, 12–16)
top_axes_positions = [(0, slice(4, 8)), (0, slice(8, 12)), (0, slice(12, 16))]
# Bottom row (row 1): 4 subplots, each 4 columns wide, centered (columns 2–6, 6–10, 10–14, 14–18)
bottom_axes_positions = [
    (2, slice(2, 6)),
    (2, slice(6, 10)),
    (2, slice(10, 14)),
    (2, slice(14, 18)),
]

axes_positions = top_axes_positions + bottom_axes_positions

# For a common legend later, save the first axis's legend handles
ax_first = None

# Loop over each dataset and create a radar plot in its assigned subplot
for i, dataset in enumerate(datasets):
    r, s = axes_positions[i]
    ax = fig.add_subplot(gs[r, s], projection="polar")

    # Make the subplot larger
    ax.set_position(
        ax.get_position().expanded(1.4, 1.4)
    )  # Scale up both width and height by 40%

    # Filter data for the current dataset and only include the desired methods
    df_dataset = df_all[df_all["Dataset"] == dataset].copy()
    df_dataset = df_dataset[df_dataset["Method"].isin(METHODS_TO_INCLUDE)]
    df_dataset["Method"] = df_dataset["Method"].map(METHOD_NAMES_MAP)
    df_dataset["Method"] = pd.Categorical(
        df_dataset["Method"],
        categories=[METHOD_NAMES_MAP[m] for m in METHODS_TO_INCLUDE],
        ordered=True,
    )
    df_dataset = df_dataset.sort_values("Method")

    # Safety checks
    if (
        df_dataset["DB Memory (MB)"].isna().any()
        or (df_dataset["DB Memory (MB)"] <= 0).any()
    ):
        raise ValueError(
            f"DB Memory column contains missing or invalid values for dataset {dataset}"
        )
    if (
        df_dataset["Matching Latency (ms)"].isna().any()
        or (df_dataset["Matching Latency (ms)"] <= 0).any()
    ):
        raise ValueError(
            f"Matching Latency column contains missing or invalid values for dataset {dataset}"
        )
    if (
        df_dataset["Accuracy (R@1)"].isna().any()
        or (df_dataset["Accuracy (R@1)"] < 0).any()
    ):
        raise ValueError(
            f"Accuracy column contains missing or invalid values for dataset {dataset}"
        )

    # Compute combined metrics
    df_dataset["Total Memory (MB)"] = (
        df_dataset["Model Memory (MB)"] + df_dataset["DB Memory (MB)"]
    )
    df_dataset["Total Latency (ms)"] = (
        df_dataset["Extraction Latency GPU (ms)"] + df_dataset["Matching Latency (ms)"]
    )

    # Normalize each metric so that 0 corresponds to worst and 1 to best performance
    df_norm = df_dataset.copy()
    for m in metrics_original:
        if m == "Accuracy (R@1)":
            # Keep R@1 as is (already in 0-1 range)
            df_norm[m] = df_norm[m] / 100
        else:
            x_min = df_norm[m].min()
            x_max = df_norm[m].max()
            if metric_directions[m] == "max":
                df_norm[m] = (df_norm[m] - x_min) / (x_max - x_min)
            else:
                df_norm[m] = 1 - ((df_norm[m] - x_min) / (x_max - x_min))

    # Set up the radar chart: with 3 metrics, compute angles
    N = len(metrics_original)
    # Modify angles to put R@1 at top (90°), Memory at left (210°), and Latency at right (330°)
    angles = np.array([90, 210, 330]) * np.pi / 180

    # Plot radar polygons for each method
    for method in df_norm["Method"].unique():
        row = df_norm[df_norm["Method"] == method].iloc[0]
        values = row[metrics_original].values
        # Close the polygon by appending the first value
        values = np.concatenate((values, [values[0]]))
        plot_angles = np.concatenate((angles, [angles[0]]))
        ax.plot(plot_angles, values, label=method, linewidth=2)
        ax.fill(plot_angles, values, alpha=0.1)

    ax.set_ylim(0, 1)
    ax.set_thetagrids(angles * 180 / np.pi, labels=metrics_final, fontsize=12)
    ax.set_title(dataset, fontsize=18, pad=25)
    ax.set_yticklabels([])

    if ax_first is None:
        ax_first = ax

# Create a common legend from the first subplot's handles
handles, labels = ax_first.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, 0.02),  # Changed from 0.04 to 0.02 to move legend lower
    fontsize=18,
    ncol=len(METHODS_TO_INCLUDE),
    bbox_transform=fig.transFigure,
)

# plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig("figures/fig2.jpg", dpi=300, bbox_inches="tight")