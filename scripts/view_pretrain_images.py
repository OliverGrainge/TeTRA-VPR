import os 
import sys 
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.train.DistillDataset import JPGDataset



train_dataset_dir = "/home/oliver/datasets_drive/vpr_datasets/sf_xl/raw/panoramas"


train_dataset = JPGDataset(train_dataset_dir)

for i in range(5):
    idx = np.random.randint(0, len(train_dataset))
    image = train_dataset[idx]
    image.show()
