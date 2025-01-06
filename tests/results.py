import pandas as pd

BACKBONES = ["vit_small_PLRBitLinear"]
AGG = ["salad", "gem"]
RES = ["322", "224"]

df = pd.read_csv("data/results.csv")


for backbone in BACKBONES:
    for agg in AGG:
        for res in RES:
            id = f"{backbone}_{agg}_{res}"
            if not id in df["id"].values:
                print(f"Model {id} not found in results.csv")
