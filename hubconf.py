from models.helper import get_model
import torch.nn as nn
import torch
import os
import zipfile
import sys
from urllib.request import urlretrieve
from tqdm import tqdm

WEIGHTS_URL = "https://github.com/OliverGrainge/TeTRA-VPR/releases/download/V1.0/tetra_weights.zip"


def download_with_progress(url: str, dst_path: str):
    """
    Download a file from `url` to `dst_path`, showing a progress bar using tqdm.
    """
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=os.path.basename(dst_path)) as t:
        urlretrieve(url, dst_path, reporthook=t.update_to)


def TeTRA(aggregation_arch: str = "boq", pretrained: bool = True) -> nn.Module:
    aggregation_arch = aggregation_arch.lower()
    assert aggregation_arch in ['boq', 'salad'], (
        f"Unknown aggregation_arch '{aggregation_arch}'; only 'boq' and 'salad' are supported."
    )

    # instantiate the model on CPU (user can .to(device) afterwards)
    model = get_model(
        backbone_arch="ternaryvitbase",
        agg_arch=aggregation_arch,
        image_size=[322, 322],
    ).to("cpu")

    if pretrained:
        # where to cache
        cache_dir = torch.hub.get_dir()
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "tetra_weights.zip")

        # 1) download with progress if the zip is missing or corrupted
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1024:
            download_with_progress(WEIGHTS_URL, zip_path)

        # 2) extract into cache_dir
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(cache_dir)

        # 3) locate the directory containing weights
        default_dir = os.path.join(cache_dir, "tetra_weights")
        weight_dir = default_dir if os.path.isdir(default_dir) else cache_dir

        # 4) build expected weight path
        weight_path = os.path.join(weight_dir, f"{aggregation_arch}.pth")

        # 5) fallback: search recursively if not found
        if not os.path.isfile(weight_path):
            for root, _, files in os.walk(weight_dir):
                if f"{aggregation_arch}.pth" in files:
                    weight_path = os.path.join(root, f"{aggregation_arch}.pth")
                    break

        # final sanity check
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Couldn't find expected weights at: {weight_path}")

        # 6) load weights
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    return model
