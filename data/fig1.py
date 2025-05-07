import os

import matplotlib.pyplot as plt
import pandas as pd

# --- Unified Style for IROS ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        # "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "text.usetex": False,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "text.color": "black",  # Ensure all text is black
        "axes.labelcolor": "black",  # Ensure axis labels are black
        "axes.edgecolor": "black",  # Ensure axis edges are black
        "xtick.color": "black",  # Ensure tick labels are black
        "ytick.color": "black",  # Ensure tick labels are black
    }
)

DATASET = "Tokyo247"
df = pd.read_csv("results.csv")
df = df[df["Dataset"] == DATASET].copy()

# Filter out unwanted methods
mask = ~(
    df["Method"].str.contains("GeM")
    | df["Method"].str.contains("ResNet50-BoQ")
    | (df["Method"] == "MixVPR")
    | df["Method"].str.contains("INT8")
    | df["Method"].str.contains("DinoV2")
    | df["Method"].str.contains("TeTRA-GeM")
)
df = df[mask].copy()

# Define markers and colors for specific models
model_styles = {
    "CosPlaces": {"marker": "s", "label": "CosPlaces", "color": "#2F5597"},
    "TeTRA-MixVPR": {"marker": "^", "label": "TeTRA-MixVPR", "color": "#C00000"},
    "TeTRA-BoQ": {"marker": "v", "label": "TeTRA-BoQ", "color": "#FF0000"},
    "TeTRA-Salad": {"marker": "<", "label": "TeTRA-Salad", "color": "#FF6B6B"},
    "EigenPlaces": {"marker": "D", "label": "EigenPlaces", "color": "#548235"},
    "DinoV2": {"marker": "P", "label": "DinoV2", "color": "#7030A0"},
}


# Helper function to determine the style for a given method
def get_style(method):
    for key, style in model_styles.items():
        if key == method:  # Changed from 'in' to '==' for exact matching
            return style
    return {"marker": "o", "label": method, "color": "#808080"}


# Helper function to determine the base model name
def get_base_model(method):
    if "CosPlaces" in method:
        return "CosPlaces"
    elif "TeTRA-MixVPR" in method:
        return "TeTRA-MixVPR"
    elif "TeTRA-BoQ" in method:
        return "TeTRA-BoQ"
    elif "TeTRA-SALAD" in method:
        return "TeTRA-SALAD"
    elif "EigenPlaces" in method:
        return "EigenPlaces"
    elif "DinoV2" in method:
        return "DinoV2"
    return method


# Helper function to add annotations for descriptor sizes
def add_descriptor_annotations(
    ax, x, y, desc_size, offset=(10, 10), edge_color="gray", fontsize=14
):
    ax.annotate(
        f"{desc_size}D",
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color="black",  # Ensure annotation text is black
        bbox=dict(
            facecolor=edge_color, edgecolor=edge_color, alpha=0.5, pad=2
        ),  # Increased alpha
        arrowprops=dict(arrowstyle="->", color=edge_color, alpha=0.8),
    )  # Increased arrow alpha


# Create the figure with two subplots - side by side layout
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(22, 5)
)  # Changed to 1 row, 2 columns and adjusted figure size
fig.subplots_adjust(wspace=0.1)  # Adjust spacing between plots

# Update the suptitle position and layout
fig.suptitle(
    f"Recall@1 vs. Matching Memory and Latency Trade-offs on {DATASET}",
    fontsize=18,
    y=1.0,
)  # Moved title up slightly

# --- Plot 1: Database Memory vs Accuracy (Line Plot with Markers) ---
df["BaseModel"] = df["Method"].apply(get_base_model)
for base_model, group in df.groupby("BaseModel"):
    style = get_style(base_model)
    group = group.sort_values("DB Memory (MB)")
    line = ax1.plot(
        group["DB Memory (MB)"],
        group["Accuracy (R@1)"],
        marker=style["marker"],
        markersize=10,
        linewidth=3.0,
        color=style["color"],
        label=style["label"],
        alpha=0.9,
    )

    # Add descriptor size annotations for each point
    for idx, row in group.iterrows():
        descriptor_size = None
        offset = (10, 10)
        edge_color = "grey"
        if row["Method"] == "CosPlaces-D32":
            descriptor_size = 32
            offset = (10, 20)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "CosPlaces-D1024":
            descriptor_size = 1024
            offset = (10, -30)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "EigenPlaces-D256":
            descriptor_size = 256
            offset = (10, -30)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "TeTRA-BoQ-DD[1]":
            descriptor_size = 12288
            offset = (20, -60)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if descriptor_size is not None:
            add_descriptor_annotations(
                ax1,
                row["DB Memory (MB)"],
                row["Accuracy (R@1)"],
                descriptor_size,
                offset,
                edge_color,
            )

ax1.set_xlabel("Database Memory (MB)", fontsize=16)
ax1.set_ylabel("Accuracy R@1 (%)", fontsize=16)
ax1.grid(True, alpha=0.4, color="gray")  # Increased alpha and specified color
for spine in ax1.spines.values():
    spine.set_linewidth(1.0)  # Increase border line width

# --- Plot 2: Matching Latency vs Accuracy (Line Plot with Markers) ---
for base_model, group in df.groupby("BaseModel"):
    style = get_style(base_model)
    group = group.sort_values("Matching Latency (ms)")
    line = ax2.plot(
        group["Matching Latency (ms)"],
        group["Accuracy (R@1)"],
        marker=style["marker"],
        markersize=10,
        linewidth=3.0,
        color=style["color"],
        label=style["label"],
        alpha=0.9,
    )

    # Add descriptor size annotations for each point
    for idx, row in group.iterrows():
        # Replace this condition with your actual data
        print(row["Method"])
        descriptor_size = None
        offset = (10, 10)
        edge_color = "grey"
        if row["Method"] == "CosPlaces-D32":
            descriptor_size = 32
            offset = (10, 20)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "CosPlaces-D1024":
            descriptor_size = 1024
            offset = (10, -30)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "EigenPlaces-D256":
            descriptor_size = 256
            offset = (10, -30)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if row["Method"] == "TeTRA-BoQ-DD[1]":
            descriptor_size = 12288
            offset = (20, -60)
            edge_color = model_styles[row["BaseModel"]]["color"]
        if descriptor_size is not None:
            add_descriptor_annotations(
                ax2,
                row["Matching Latency (ms)"],
                row["Accuracy (R@1)"],
                descriptor_size,
                offset,
                edge_color,
            )

ax2.set_xlabel("Matching Latency (ms)", fontsize=16)
ax2.set_ylabel("Accuracy R@1 (%)", fontsize=16)
ax2.grid(True, alpha=0.4, color="gray")  # Increased alpha and specified color
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)  # Increase border line width

# Remove the combined legend creation and instead add legends to each subplot
ax1.legend(
    loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    fontsize=12,
    frameon=True,
    framealpha=1.0,
    edgecolor="black",
    borderpad=1.2,
)

ax2.legend(
    loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    fontsize=12,
    frameon=True,
    framealpha=1.0,
    edgecolor="black",
    borderpad=1.2,
)

# Adjust the subplot spacing
fig.subplots_adjust(right=0.95, top=0.85)

# Save the figure
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig1.jpg", dpi=300, bbox_inches="tight")
plt.show()
