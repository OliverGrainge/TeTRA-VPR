import pandas as pd

df = pd.read_csv("results.csv")

columns = [
    "MSLS",
    "Pitts30k",
    "Tokyo247",
    "SVOX-night",
    "SVOX-rain",
    "SVOX-snow",
    "SVOX-sun",
]
methods = [
    "DinoV2-Salad",
    "DinoV2-BoQ",
    "ResNet50-BoQ",
    "EigenPlaces-D256",
    "EigenPlaces-D512",
    "EigenPlaces-D2048",
    "CosPlaces-D128",
    "CosPlaces-D512",
    "CosPlaces-D2048",
    "DinoV2-Salad-INT8",
    "DinoV2-BoQ-INT8",
    "ResNet50-BoQ-INT8",
    "EigenPlaces-D2048-INT8",
    "CosPlaces-D2048-INT8",
    "TeTRA-BoQ-DD[1]",
    "TeTRA-BoQ-DD[2]",
    "TeTRA-SALAD-DD[1]",
    "TeTRA-SALAD-DD[2]",
    "TeTRA-GeM-DD[1]",
    "TeTRA-GeM-DD[2]",
    "TeTRA-MixVPR-DD[1]",
    "TeTRA-MixVPR-DD[2]",
]

pivot_df_memory = df.pivot(index="Method", columns="Dataset", values="DB Memory (MB)")

pivot_df_model_memory = df.pivot(
    index="Method", columns="Dataset", values="Model Memory (MB)"
)

# Select only the relevant columns and create a pivot table
pivot_df = df.pivot(index="Method", columns="Dataset", values="Accuracy (R@1)")

filtered_df = pivot_df.loc[methods, columns]

filtered_df_memory = pivot_df_memory.loc[methods, columns]
filtered_df_model_memory = pivot_df_model_memory.loc[methods, columns]

# Calculate total memory (DB + Model) for each method-dataset combination
total_memory = filtered_df_memory.add(filtered_df_model_memory)

# Calculate memory efficiency using total memory
filtered_df_memory_efficiency = filtered_df / total_memory

filtered_df_memory_efficiency = filtered_df_memory_efficiency.rename(
    columns={"MSLS": "MSLS/MB", "Pitts30K": "Pitts30K/MB", "Tokyo247": "Tokyo247/MB"}
)

df = pd.concat([filtered_df, filtered_df_memory_efficiency], axis=1)

# Convert to LaTeX table with formatting
latex_table = df.to_latex(
    float_format=lambda x: "{:.1f}".format(x),
    caption="R@1 Accuracy comparison across datasets",
    label="tab:accuracy_comparison",
    escape=False,
)

# Modify the table environment to span two columns
latex_table = latex_table.replace("\\begin{table}", "\\begin{table*}").replace(
    "\\end{table}", "\\end{table*}"
)

print(latex_table)
