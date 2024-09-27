import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib.pyplot as plt
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu 
from pytorch_metric_learning.miners.base_miner import BaseMiner



class MultiSimilarityMiner(BaseMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_to_recordable_attributes(name="epsilon", is_stat=False)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.tensor([], device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]

        return a1, p, a2, n

    def get_default_distance(self):
        return CosineSimilarity()
# Set random seed for reproducibility
torch.manual_seed(0)

# Define the number of classes and the number of embeddings per class
num_classes = 2
num_samples_per_class = 1000
embedding_dim = 1024

# Generate random embeddings for each class
embeddings = []
labels = []
for class_idx in range(num_classes):
    # Create embeddings from a normal distribution for each class
    class_embeddings = torch.randn(num_samples_per_class, embedding_dim) + class_idx * 5
    class_labels = torch.full((num_samples_per_class,), class_idx)
    embeddings.append(class_embeddings)
    labels.append(class_labels)

# Stack embeddings and labels into tensors
embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)

# Visualize the embedding distribution
#plt.figure(figsize=(6, 6))
#for class_idx in range(num_classes):
#    class_embeddings = embeddings[labels == class_idx]
#    plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], label=f'Class {class_idx}', s=50)
#plt.legend()
# #plt.title("Embedding Distributions of Different Classes")
#plt.show()

# Initialize the MultiSimilarityMiner
miner = MultiSimilarityMiner(epsilon=0.0, distance=CosineSimilarity())
# Use the miner to find positive and negative pairs
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
miner_outputs = miner(embeddings, labels)


print(miner_outputs[0].shape, miner_outputs[1].shape, miner_outputs[2].shape, miner_outputs[3].shape)