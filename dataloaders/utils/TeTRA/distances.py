import torch
from pytorch_metric_learning.distances import BaseDistance, CosineSimilarity
from pytorch_metric_learning.losses import MultiSimilarityLoss
from torch.autograd import Function


class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # Save input_tensor for use in backward
        ctx.save_for_backward(input_tensor)

        # Binarize input_tensor: +1 if x > 0, -1 otherwise
        output = torch.where(
            input_tensor >= 0,
            torch.ones_like(input_tensor),
            -torch.ones_like(input_tensor),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors

        # Straight-Through Estimator (STE) for binarization
        # Gradient is passed through if the weight is in the range (-1, 1)
        grad_input = grad_output.clone()
        grad_input[input_tensor.abs() > 1] = 0
        return grad_input


def binarize(input_tensor):
    return BinarizeSTE.apply(input_tensor)


class HammingDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=False, is_inverted=False, **kwargs)

    def compute_mat(self, query_emb, ref_emb):
        query_emb, ref_emb = binarize(query_emb), binarize(ref_emb)
        dist = (query_emb.shape[1] - torch.matmul(query_emb, ref_emb.t())) / 2
        return ((dist / query_emb.shape[1]) * 2) - 1

    def pairwise_distance(self, query_emb, ref_emb):
        query_emb, ref_emb = binarize(query_emb), binarize(ref_emb)
        # Compute the dot product between all queries and all references
        dot_product = torch.matmul(query_emb, ref_emb.t())
        # Calculate Hamming distance using the formula
        dist = (query_emb.shape[1] - dot_product) / 2
        return ((dist / query_emb.shape[1]) * 2) - 1
