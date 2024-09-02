import torch
from pytorch_metric_learning import losses
import torch.nn.functional as F

from pytorch_metric_learning.distances import BaseDistance


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)
    


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings



class HammingDistance(BaseDistance):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        self.temperature = temperature
    
    def compute_mat(self, query_emb, ref_emb):
        # Ensure the embeddings are in the range [0, 1] and represent soft binary values
        abs_diff = torch.abs(query_emb.unsqueeze(1) - ref_emb.unsqueeze(0))
        abs_diff_normed = abs_diff.sum(dim=2) / query_emb.shape[1]
        smooth_hamming_similarity = 1 - abs_diff_normed
        return smooth_hamming_similarity

    def pairwise_distance(self, query_emb, ref_emb):
        abs_diff = torch.abs(query_emb - ref_emb)
        abs_diff_normed = abs_diff.sum(dim=1) / query_emb.shape[1]
        smooth_hamming_similarity = 1 - abs_diff_normed
        return smooth_hamming_similarity





# Test script
def test_similarity_classes(query_emb, ref_emb):
    # Test CosineSimilarity
    cosine_similarity = CosineSimilarity()
    
    # Since we need normalized embeddings for cosine similarity, normalize manually for testing
    query_emb_normalized = F.normalize(query_emb, p=2, dim=1)
    ref_emb_normalized = F.normalize(ref_emb, p=2, dim=1)
    
    cosine_similarity_matrix = cosine_similarity.compute_mat(query_emb_normalized, ref_emb_normalized)
    cosine_similarity_pairwise = cosine_similarity.pairwise_distance(query_emb_normalized, ref_emb_normalized)
    
    #print("CosineSimilarity - Similarity Matrix:\n", cosine_similarity_matrix)
    print("CosineSimilarity - Pairwise Similarity:\n", cosine_similarity_pairwise)


# Test the HammingSimilarity class
def test_hamming_similarity(query_emb, ref_emb):
    # Generate some random binary embeddings

    # Instantiate HammingSimilarity
    hamming_similarity = HammingDistance()
    
    # Compute similarity matrix
    hamming_similarity_matrix = hamming_similarity.compute_mat(query_emb, ref_emb)
    #print("HammingSimilarity - Similarity Matrix:\n", hamming_similarity_matrix)
    
    # Compute pairwise similarity
    hamming_pairwise_similarity = hamming_similarity.pairwise_distance(query_emb, ref_emb)
    print("HammingSimilarity - Pairwise Similarity:\n", hamming_pairwise_similarity)


def test_multi_similarity_loss(query_emb, ref_emb, labels, distance_metric):
    loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=distance_metric)
    
    # Combine query and reference embeddings into one tensor
    embeddings = torch.cat((query_emb, ref_emb))
    
    # Compute the loss
    loss = loss_func(embeddings, labels)
    print(f"Multi-Similarity Loss with {distance_metric.__class__.__name__}: {loss.item()}")

if __name__ == "__main__":
    # Create example embeddings and labels
    query_emb = torch.tensor([[0.5, 0.5]])
    ref_emb = torch.tensor([[5.0, 0.5]])
    labels = torch.tensor([1, 1])  # Assuming first two are the same class, and last two are another class
    
    # Test with different similarity measures
    test_multi_similarity_loss(query_emb, ref_emb, labels, CosineSimilarity())
    test_multi_similarity_loss(query_emb, ref_emb, labels, HammingDistance())