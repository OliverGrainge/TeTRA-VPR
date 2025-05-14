import argparse

import faiss
import numpy as np
import torch


def _parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--img_height", type=int, required=True, help="Image height")
    parser.add_argument("--img_width", type=int, required=True, help="Image width")
    return parser.parse_args()


def _freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def _get_image(image_height: int, image_width: int):
    return torch.randn(1, 3, image_height, image_width)


def _get_desc_size(model, image):
    return model(image).shape[1]


def _load_model(model_name: str):
    if model_name == "tetra":
        model = torch.hub.load(
            repo_or_dir="OliverGrainge/TeTRA-VPR",
            model="TeTRA",
            aggregation_arch="BoQ",
            pretrained=True,
        )
    elif model_name == "cosplace":
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )

    elif model_name == "dinoboq":
        model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
        )
        old_forward = model.forward
        model.forward = lambda x: old_forward(x)[0]
    else:
        raise ValueError(f"Model {model_name} not found")
    return _freeze_model(model)


def float_to_binary_desc(desc: np.ndarray) -> np.ndarray:
    """Convert float descriptors to binary packed format."""
    binary = (desc > 0).astype(np.bool_)
    n_bytes = (binary.shape[1] + 7) // 8
    return np.packbits(binary, axis=1)[:, :n_bytes]


def _get_index(model_name: str, desc_size: int, dataset_size: int):
    if model_name == "tetra":
        index = faiss.IndexBinaryFlat(desc_size * 8)
        dbvec = np.random.randn(dataset_size, desc_size * 8).astype(np.float32)
        assert dbvec.shape[1] % 8 == 0, "Descriptor size must be divisible by 8"
        dbvec = float_to_binary_desc(dbvec)
        dbvec = np.array(dbvec).astype(np.uint8)
        index.add(dbvec)
    else:
        index = faiss.IndexFlatIP(desc_size)
        dbvec = np.random.randn(dataset_size, desc_size)
        index.add(dbvec)
    return index


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = _load_model(args.model_name)
    model.to(device)
    x = _get_image(args.img_height, args.img_width)
    x = x.to(device)
    desc_size = _get_desc_size(model, x)
    index = _get_index(args.model_name, desc_size, args.dataset_size)

    print(f"Starting speed measurement for {args.model_name}")
    print(
        f"Image size: {args.img_height}x{args.img_width}, Dataset size: {args.dataset_size}"
    )

    num_iterations = 100
    total_fps = 0
    total_latency = 0
    total_inference_latency = 0
    total_matching_latency = 0
    desc_dtype = np.uint8 if args.model_name == "tetra" else np.float32

    # Warmup
    for _ in range(10):
        desc = model(x).cpu().numpy()
        desc = desc.astype(desc_dtype)
        index.search(desc, 1)

    for i in range(num_iterations):
        # Measure feature extraction time
        start_time = time.time()
        desc = model(x).cpu().numpy()
        feature_time = time.time() - start_time
        inference_latency = feature_time * 1000  # Convert to ms

        # Measure search time
        desc = desc.astype(desc_dtype)
        start_time = time.time()
        index.search(desc, 1)
        search_time = time.time() - start_time
        matching_latency = search_time * 1000  # Convert to ms

        # Total processing time
        total_time = feature_time + search_time
        latency = total_time * 1000  # Convert to ms
        fps = 1.0 / total_time

        total_fps += fps
        total_latency += latency
        total_inference_latency += inference_latency
        total_matching_latency += matching_latency

        print(
            f"Iteration {i+1}/{num_iterations}: Latency = {latency:.2f} ms, FPS = {fps:.2f}, Inference = {inference_latency:.2f} ms, Matching = {matching_latency:.2f} ms"
        )

    avg_fps = total_fps / num_iterations
    avg_latency = total_latency / num_iterations
    avg_inference_latency = total_inference_latency / num_iterations
    avg_matching_latency = total_matching_latency / num_iterations

    print("\n===== Results =====")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Inference Latency: {avg_inference_latency:.2f} ms")
    print(f"Average Matching Latency: {avg_matching_latency:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Feature Extraction Size: {desc_size}")


if __name__ == "__main__":
    import time

    args = _parseargs()
    main(args)
