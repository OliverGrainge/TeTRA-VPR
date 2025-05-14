import argparse
import os
from glob import glob

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_dir", type=str, required=True)
    args.add_argument("--database_dir", type=str, default="database")
    args.add_argument("--queries_dir", type=str, default="queries")
    args.add_argument("--dataset_name", type=str, required=True)
    args.add_argument("--split", type=str, required=True)
    args.add_argument("--truncate", type=str, default="")
    return args.parse_args()


def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [
            p
            for p in images_paths
            if os.path.isfile(p)
            and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        if len(images_paths) == 0:
            raise FileNotFoundError(
                f"Directory {dataset_folder} does not contain any images"
            )
    return images_paths


def _positive_match_idxs(gt):
    idxs = []
    for i, positives in enumerate(gt):
        if len(positives) > 0:
            idxs.append(i)
    return idxs


def _compute_gt(database_paths, queries_paths):
    try:
        # This is just a sanity check
        image_path = database_paths[0]
        utm_east = float(image_path.split("@")[1])
        utm_north = float(image_path.split("@")[2])
    except:
        raise ValueError(
            "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
            f"but it is {image_path}, which does not contain the UTM coordinates."
        )

    database_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in database_paths]
    ).astype(float)
    queries_utms = np.array(
        [(path.split("@")[1], path.split("@")[2]) for path in queries_paths]
    ).astype(float)

    # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    gt = knn.radius_neighbors(queries_utms, radius=25, return_distance=False)

    # Filter out queries that don't have any positives
    return gt


def compute_paths(dataset_dir, database_dir=None, queries_dir=None):
    database_folder = (
        os.path.join(dataset_dir, "database")
        if database_dir is None
        else os.path.join(dataset_dir, database_dir)
    )
    queries_folder = (
        os.path.join(dataset_dir, "queries")
        if queries_dir is None
        else os.path.join(dataset_dir, queries_dir)
    )
    database_paths = read_images_paths(database_folder)
    queries_paths = read_images_paths(queries_folder)
    gt = _compute_gt(database_paths, queries_paths)

    # positive_match_idxs = _positive_match_idxs(gt)
    # queries_paths = np.array([queries_paths[i] for i in positive_match_idxs])
    # gt = np.array([gt[i] for i in positive_match_idxs], dtype=object)
    return database_paths, queries_paths, gt


def _truncate_paths(paths, root_dir, dataset_name):
    trunc_str = root_dir[: root_dir.find(dataset_name) + len(dataset_name)]
    new_paths = [pth.replace(trunc_str, "") for pth in paths]
    return new_paths


if __name__ == "__main__":
    args = _parse_args()
    database_paths, queries_paths, gt = compute_paths(
        args.dataset_dir, args.database_dir, args.queries_dir
    )
    database_paths = _truncate_paths(database_paths, args.dataset_dir, args.truncate)
    queries_paths = _truncate_paths(queries_paths, args.dataset_dir, args.truncate)
    os.makedirs("image_paths", exist_ok=True)
    print(f"========== {args.dataset_name} ==========")
    print("Sample database path: ", database_paths[0])
    print("Sample query path: ", queries_paths[0])
    np.save(f"image_paths/{args.dataset_name}_{args.split}_qImages.npy", queries_paths)
    np.save(
        f"image_paths/{args.dataset_name}_{args.split}_dbImages.npy", database_paths
    )
    np.save(f"image_paths/{args.dataset_name}_{args.split}_gt.npy", gt)
