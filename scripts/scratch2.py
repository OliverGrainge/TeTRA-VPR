
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 

model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
model.eval()
model = model.cuda()

base_transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

val_ds = PittsburghDataset(input_transform=base_transform)
#val_ds = SPEDDataset(input_transform=base_transform)
dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)


descriptors = torch.zeros(len(val_ds), 2048)
for images, index in tqdm(dl): 
    images = images.cuda()
    desc = model(images)
    descriptors[index] = desc.detach().cpu()


query_desc = descriptors[:val_ds.num_references].numpy().astype(np.float32)
map_desc = descriptors[val_ds.num_references:].numpy().astype(np.float32)
gt = val_ds.ground_truth




faiss.normalize_L2(query_desc)
index = faiss.IndexFlatIP(query_desc.shape[1]) 
faiss.normalize_L2(map_desc)
index.add(map_desc)


all_distances, all_predictions = index.search(query_desc, len(map_desc))
topk_distances, topk_predictions = index.search(query_desc, 10)

all_gt = []
topk_gt = []
for query_idx in range(len(val_ds.ground_truth)): 
    all_correct = []
    for pred in all_predictions[query_idx]: 
        if pred in val_ds.ground_truth[query_idx]: 
            all_correct.append(1)
        else: 
            all_correct.append(0)
    all_gt.append(all_correct)


    topk_correct = []
    for pred in topk_predictions[query_idx]: 
        if pred in val_ds.ground_truth[query_idx]: 
            topk_correct.append(1)
        else: 
            topk_correct.append(0)
    topk_gt.append(topk_correct)

all_gt = np.array(all_gt)
topk_gt = np.array(topk_gt)

print(all_gt.shape, all_distances.shape)
all_distances_correct = all_distances[all_gt.astype(bool)].flatten()
all_distances_incorrect = all_distances[~all_gt.astype(bool)].flatten()
topk_distances_correct = topk_distances[topk_gt.astype(bool)].flatten()
topk_distances_incorrect = topk_distances[~topk_gt.astype(bool)].flatten()


print(all_distances_correct.shape, all_distances_incorrect.shape)
print(topk_distances_correct.shape, topk_distances_incorrect.shape)

plt.hist(all_distances_incorrect, bins=100, density=True, label="all incorrect")
plt.hist(topk_distances_incorrect, bins=100, density=True, label="topk incorrect")
plt.legend()
plt.show()


