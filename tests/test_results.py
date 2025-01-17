import sys 
import os 
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

datasets = ("Pittsburgh30k", "Tokyo247", "MSLS")
desc_divide_factors = ("1", "2", "4")
bacbkones = ("vitsmallt", "vitbaset")
agg_archs = ("boq", "salad", "gem", "mixvpr")
image_sizes = ("224", "322")

@pytest.mark.parametrize("backbone", bacbkones)
@pytest.mark.parametrize("desc_divide_factor", desc_divide_factors)
@pytest.mark.parametrize("agg_arch", agg_archs)
@pytest.mark.parametrize("image_size", image_sizes)
def test_results_completeness(backbone, desc_divide_factor, agg_arch, image_size):
    df = pd.read_csv("data/results.csv")
    model_id = f"{backbone}_{agg_arch}_{image_size}_DescDividerFactor[{desc_divide_factor}]"
    # Check if model_id exists in df["id"]
    if model_id not in df["id"].values:
        print(f"Missing model_id: {model_id}")
    else:
        # Check if results exist for each dataset
        for dataset in datasets:
            if f"{dataset}_cosine_R@1" not in df.columns or df.loc[df["id"] == model_id, f"{dataset}_cosine_R@1"].isnull().all():
                print(f"Missing results for {model_id} in dataset: {dataset}_cosine_R@1")
            if f"{dataset}_hamming_R@1" not in df.columns or df.loc[df["id"] == model_id, f"{dataset}_hamming_R@1"].isnull().all():
                print(f"Missing results for {model_id} in dataset: {dataset}_hamming_R@1")
            print(df[df["id"] == model_id][f"{dataset}_cosine_R@1"])
