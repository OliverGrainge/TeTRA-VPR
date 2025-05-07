from typing import Union

import torchvision.transforms as T
import matplotlib.pyplot as plt
import time

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}



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
                    p=0.9,  # Increased probability for more occlusion robustness
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
                    p=1.0,  # Moderate probability
                    scale=(0.2, 0.5),  # Slightly smaller regions
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




if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage

    image_path = "/Users/olivergrainge/Documents/PaperReviews/TeTRA/figures/example_place.jpg" 

    img = Image.open(image_path)
    img = img.resize((320, 320))
    img.show()
    """

    transform = _get_augmentation("severe", 320)
    print(transform)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    plt.show()
    time.sleep(1)  # Wait 1 second

    # Plot transformed images one by one
    for i in range(9):  # 9 augmented images
        img_t = transform(img)
        # Convert tensor to numpy for plotting (handle channel dimension and normalization)
        img_np = img_t.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
        img_np = (img_np * IMAGENET_MEAN_STD['std'] + IMAGENET_MEAN_STD['mean']).clip(0, 1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.title(f'Augmentation {i+1}')
        plt.axis('off')
        plt.show()
        time.sleep(1)  # Wait 1 second between images
    """
