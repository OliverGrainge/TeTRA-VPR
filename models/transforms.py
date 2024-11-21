import torchvision.transforms as T
from typing import Union

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

def _get_preset_transform(preset):
    if preset.lower() == "dinov2_boq":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (322, 322),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset.lower() == "resnet50_boq":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (384, 384),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset.lower() == "dinosalad":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (322, 322),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset.lower() == "eigenplaces":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif preset.lower() == "cosplaces":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")
    return transform




def _get_augmentation(augmentation_level: str, image_size: Union[tuple, int]):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if augmentation_level.lower() == "severe":
        return T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.85, 1.0)
                ),  # Randomly crop and resize the image
                T.ColorJitter(
                    brightness=0.5, contrast=0.4, saturation=0.2, hue=0.1
                ),  # Randomly change brightness, contrast, etc.
                T.GaussianBlur(
                    kernel_size=(3, 7), sigma=(0.1, 1.0)
                ),  # Apply Gaussian blur
                T.ToTensor(),  # Convert image to tensor
                T.RandomErasing(
                    p=0.2,  # Increased probability for more occlusion robustness
                    scale=(0.02, 0.15),  # Smaller regions (like pedestrians/cars)
                    ratio=(0.3, 2.0),  # More varied aspect ratios
                    value="random",  # Random values to simulate different occlusions
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif augmentation_level.lower() == "moderate":
        return T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.9, 1.0)
                ),  # Randomly crop and resize the image
                T.ColorJitter(
                    brightness=0.35, contrast=0.2, saturation=0.1, hue=0.05
                ),  # Randomly change brightness, contrast, etc.
                T.GaussianBlur(
                    kernel_size=(3, 7), sigma=(0.1, 0.5)
                ),  # Apply Gaussian blur
                T.ToTensor(),  # Convert image to tensor
                T.RandomErasing(
                    p=0.2,  # Moderate probability
                    scale=(0.02, 0.10),  # Slightly smaller regions
                    ratio=(0.3, 2.0),
                    value="random",
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    elif augmentation_level.lower() == "light":
        return T.Compose(
            [
                T.Resize(image_size),
                T.ColorJitter(
                    brightness=0.3, contrast=0.1, saturation=0.1, hue=0.05
                ),  # Randomly change brightness, contrast, etc.
                T.ToTensor(),  # Convert image to tensor
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    elif augmentation_level.lower() == "none":
        return T.Compose(
            [
                T.Resize(image_size),  # Resize the image to the specified size
                T.ToTensor(),  # Convert image to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise Exception(f"Augmentation type {augmentation_level} not found")
    


def get_transform(augmentation_level: Union[str, None] = "None", image_size: Union[tuple, int] = (224, 224), preset: Union[str, None] = None): 
    if preset is not None: 
        return _get_preset_transform(preset=preset)
    else: 
        return _get_augmentation(augmentation_level=augmentation_level, image_size=image_size)


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    img_path = os.path.join(os.path.dirname(__file__), "assets/example_image.jpg")
    # Load image
    img = Image.open(img_path)

    # Get all three augmentation transforms
    light_aug = get_transform("LightAugment", (384, 384))
    moderate_aug = get_transform("ModerateAugment", (384, 384))
    severe_aug = get_transform("SevereAugment", (384, 384))
    no_aug = get_transform("NoAugment", (384, 384))

    # Number of examples per augmentation type
    n_examples = 10

    # Apply multiple augmentations
    augmented_light = [light_aug(img) for _ in range(n_examples)]
    augmented_moderate = [moderate_aug(img) for _ in range(n_examples)]
    augmented_severe = [severe_aug(img) for _ in range(n_examples)]
    augmented_none = [no_aug(img) for _ in range(n_examples)]

    # Convert tensors back to images for display
    def tensor_to_display(tensor):
        mean = torch.tensor(IMAGENET_MEAN_STD["mean"]).view(3, 1, 1)
        std = torch.tensor(IMAGENET_MEAN_STD["std"]).view(3, 1, 1)
        img = tensor * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        return img.numpy()

    # Convert all augmented images
    augmented_light = [tensor_to_display(x) for x in augmented_light]
    augmented_moderate = [tensor_to_display(x) for x in augmented_moderate]
    augmented_severe = [tensor_to_display(x) for x in augmented_severe]
    augmented_none = [tensor_to_display(x) for x in augmented_none]

    plt.figure(figsize=(4 * n_examples, 12))  # Increased height to accommodate 4 rows

    # Create a 4xN grid (4 rows, N columns where N is n_examples)
    for i in range(n_examples):
        # No augmentation row
        plt.subplot(4, n_examples, i + 1)
        plt.imshow(augmented_none[i])
        plt.title(f"No Augment {i+1}" if i == 0 else str(i + 1))
        plt.axis("off")

        # Light augmentation row
        plt.subplot(4, n_examples, n_examples + i + 1)
        plt.imshow(augmented_light[i])
        plt.title(f"Light Augment {i+1}" if i == 0 else str(i + 1))
        plt.axis("off")

        # Moderate augmentation row
        plt.subplot(4, n_examples, 2 * n_examples + i + 1)
        plt.imshow(augmented_moderate[i])
        plt.title(f"Moderate Augment {i+1}" if i == 0 else str(i + 1))
        plt.axis("off")

        # Severe augmentation row
        plt.subplot(4, n_examples, 3 * n_examples + i + 1)
        plt.imshow(augmented_severe[i])
        plt.title(f"Severe Augment {i+1}" if i == 0 else str(i + 1))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

