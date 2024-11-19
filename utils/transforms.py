import torchvision.transforms as T 




IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}



def get_train_transform(augment_type: str, image_size: tuple): 
    if augment_type == "SevereAugment": 
        return T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0)
                ),  # Randomly crop and resize the image
                T.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),  # Randomly change brightness, contrast, etc.
                T.GaussianBlur(
                    kernel_size=(3, 7), sigma=(0.1, 0.5)
                ),  # Apply Gaussian blur
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),  # Convert image to tensor
                T.RandomErasing(
                    p=0.1, scale=(0.02, 0.05), ratio=(0.3, 1.7), value="random"
                ),  # Cut out random parts
                T.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )
    elif augment_type == "LightAugment": 
        return T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0)
                ),  # Randomly crop and resize the image
                T.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),  # Randomly change brightness, contrast, etc.
                T.GaussianBlur(
                    kernel_size=(3, 7), sigma=(0.1, 0.5)
                ),  # Apply Gaussian blur
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),  # Convert image to tensor
                T.RandomErasing(
                    p=0.1, scale=(0.02, 0.05), ratio=(0.3, 1.7), value="random"
                ),
                T.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )
    elif augment_type == "NoAugment": 
        return T.Compose(
            [
                T.Resize(image_size),  # Resize the image to the specified size
                T.ToTensor(),  # Convert image to tensors
                T.Normalize(
                    mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
                ),
            ]
        )




def get_val_transform(image_size: tuple): 
    return T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]
            ),
        ]
    )