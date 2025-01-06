import argparse
import glob
import os
from os.path import join

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm


def main(args):
    dataset_folder = join(args.dataset_folder, args.subset_folder)
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    #### Read paths and UTM coordinates for all images.
    database_folder = join(dataset_folder, args.database_folder)
    queries_folder = join(dataset_folder, args.queries_folder)
    if not os.path.exists(database_folder):
        raise FileNotFoundError(f"Folder {database_folder} does not exist")
    if not os.path.exists(queries_folder):
        raise FileNotFoundError(f"Folder {queries_folder} does not exist")

    database_paths = sorted(
        glob.glob(join(database_folder, "**", "*.jpg"), recursive=True)
    )
    queries_paths = sorted(
        glob.glob(join(queries_folder, "**", "*.jpg"), recursive=True)
    )

    # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
    database_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in database_paths]
    ).astype(float)
    queries_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in queries_paths]
    ).astype(float)

    # Find soft_positives_per_query, which are within val_positive_dist_threshold
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    soft_positives_per_query = knn.radius_neighbors(
        queries_utms, radius=25, return_distance=False
    )

    images_paths = list(database_paths) + list(queries_paths)
    if args.dataset_folder.endswith("/"):
        args.dataset_folder = args.dataset_folder[:-1]
    remove_str = "/".join(args.dataset_folder.split("/")[:-1])

    database_paths = np.array(
        [pth.replace(remove_str, "").lstrip("/") for pth in database_paths]
    )
    queries_paths = np.array(
        [pth.replace(remove_str, "").lstrip("/") for pth in queries_paths]
    )
    assert queries_paths.shape[0] == soft_positives_per_query.shape[0]
    os.makedirs("./image_paths", exist_ok=True)

    print("dataset_path", database_paths[0], len(database_paths))
    print("queries_path", queries_paths[0], len(queries_paths))
    print(
        "soft_positives_per_query",
        soft_positives_per_query[0],
        len(soft_positives_per_query),
    )
    np.save(
        "image_paths/" + args.dataset_name + "_" + args.split + "_dbImages.npy",
        database_paths,
    )
    np.save(
        "image_paths/" + args.dataset_name + "_" + args.split + "_qImages.npy",
        queries_paths,
    )
    np.save(
        "image_paths/" + args.dataset_name + "_" + args.split + "_gt.npy",
        soft_positives_per_query,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process UTM image paths and queries.")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Path to the datasets folder."
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Path to the datasets folder."
    )

    parser.add_argument(
        "--subset_folder", type=str, default="", help="Path to the datasets folder."
    )
    parser.add_argument(
        "--queries_folder", type=str, default="queries", help="Name of the dataset."
    )
    parser.add_argument(
        "--database_folder", type=str, default="database", help="Name of the dataset."
    )

    args = parser.parse_args()
    main(args)
