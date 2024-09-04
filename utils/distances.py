import torch
from pytorch_metric_learning import losses
import torch.nn.functional as F
from pytorch_metric_learning.distances import BaseDistance
import torch
from torch.autograd import Function

import matplotlib.pyplot as plt

class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dist = torch.matmul(query_emb, ref_emb.t())
        return dist

    def pairwise_distance(self, query_emb, ref_emb):
        dist = torch.sum(query_emb * ref_emb, dim=1)
        return dist
    


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings



class HammingDistanceSoft(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=False, is_inverted=True, **kwargs)
        assert self.is_inverted
        #assert self.normalize_embeddings
        self.temperature = 0.05
    
    def compute_mat(self, query_emb, ref_emb):
        query_emb, ref_emb = torch.tanh(query_emb/self.temperature), torch.tanh(ref_emb/self.temperature)
        return torch.matmul(query_emb, ref_emb.t()) / query_emb.shape[1]

    def pairwise_distance(self, query_emb, ref_emb):
        query_emb, ref_emb = torch.tanh(query_emb/self.temperature), torch.tanh(ref_emb/self.temperature)
        return torch.sum(query_emb * ref_emb, dim=1) / query_emb.shape[1]




#=====================================================================================================================


class BinarizeSTEsymmetric(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # Save input_tensor for use in backward
        ctx.save_for_backward(input_tensor)
        
        # Binarize input_tensor: +1 if x > 0, -1 otherwise
        output = torch.where(input_tensor >= 0, torch.ones_like(input_tensor), -torch.ones_like(input_tensor))
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        
        # Straight-Through Estimator (STE) for binarization
        # Gradient is passed through if the weight is in the range (-1, 1)
        grad_input = grad_output.clone()
        grad_input[input_tensor.abs() > 1] = 0
        
        return grad_input


class BinarizeSTEasymmetric(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # Save input_tensor for use in backward
        ctx.save_for_backward(input_tensor)
        
        # Binarize input_tensor: +1 if x > 0, 0 otherwise
        output = torch.where(input_tensor >= 0, torch.ones_like(input_tensor), torch.zeros_like(input_tensor))
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        
        # Straight-Through Estimator (STE) for binarization
        # Gradient is passed through if the weight is in the range (-1, 1)
        grad_input = grad_output.clone()
        grad_input[input_tensor.abs() > 1] = 0
        
        return grad_input

# Custom binarize function with STE
def binarize_ste(input_tensor, symmetric=True):
    if symmetric == True:
        return BinarizeSTEsymmetric.apply(input_tensor)
    else: 
        return BinarizeSTEasymmetric.apply(input_tensor)



class HammingDistanceSTE(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=False, is_inverted=True, **kwargs)
        assert self.is_inverted
        #assert self.normalize_embeddings
        self.temperature = 0.75
    
    def compute_mat(self, query_emb, ref_emb):
        query_emb, ref_emb = binarize_ste(query_emb), binarize_ste(ref_emb)
        dist = torch.matmul(query_emb, ref_emb.t()) / query_emb.shape[1]
        return dist

    def pairwise_distance(self, query_emb, ref_emb):
        query_emb, ref_emb = binarize_ste(query_emb), binarize_ste(ref_emb)
        dist = torch.sum(query_emb * ref_emb, dim=1) / query_emb.shape[1]
        return dist




