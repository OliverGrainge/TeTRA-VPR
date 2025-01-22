import glob
import os
import random
import tarfile
from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from prettytable import PrettyTable


class JPGDataset(Dataset):
    def __init__(self, data_directories, transform=None):
        # Convert single directory to list for consistent handling
        if isinstance(data_directories, str):
            data_directories = [data_directories]
        
        self.image_paths = []
        total_images = 0
        
        # Create a dictionary to store directory counts
        dir_counts = {}

        for data_directory in data_directories:
            print(f"\nScanning directory: {data_directory}")
            # Recursively find all jpg files in the directory
            for root, _, files in os.walk(data_directory):
                jpg_files = [os.path.join(root, f) for f in files if f.lower().endswith('.jpg')]
                num_images = len(jpg_files)
                if num_images > 0:
                    relative_path = os.path.relpath(root, data_directory)
                    dir_counts[f"{data_directory}/{relative_path}"] = num_images
                    self.image_paths.extend(jpg_files)
                    total_images += num_images

        # Create and configure the table
        table = PrettyTable()
        table.field_names = ["Directory", "Image Count"]
        table.align["Directory"] = "l"  # Left align directory
        table.align["Image Count"] = "r"  # Right align count
        table.max_width["Directory"] = 80  # Limit directory column width
        
        # Add rows to the table
        for directory, count in dir_counts.items():
            table.add_row([directory, f"{count:,}"])
        
        # Add total row
        table.add_row(["TOTAL", f"{total_images:,}"])
        
        print("\nDirectory Statistics:")
        print(table)

        # Add a cache for known bad images
        self.bad_images = set()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Skip known bad images
        while idx in self.bad_images and len(self.bad_images) < len(self.image_paths):
            idx = (idx + 1) % len(self.image_paths)
            
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                
                # Continue processing the image as usual
                width, height = image.size

                # crop image if it is panoramic and a random direction 
                if width > height and width > 2 * height:
                    height, height = 512, 512
                    left = random.randint(0, width - height)
                    right = left + height
                    bottom = height
                    image = image.crop((left, 0, right, bottom))

                if self.transform:
                    image = self.transform(image)

                return image
                
        except Exception as e:
            # Add to bad images cache
            self.bad_images.add(idx)
            
            # If we've found too many bad images, raise an error
            if len(self.bad_images) >= len(self.image_paths):
                raise RuntimeError("All images appear to be corrupted!")
                
            # Try the next image
            return self.__getitem__((idx + 1) % len(self.image_paths))
    


