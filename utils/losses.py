import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (CosineSimilarity,
                                               DotProductSimilarity)

from .distances import HammingDistanceSoft, HammingDistanceSTE, binarize_ste

# ==================== Greedy Hash =============================================


def greedy_hash_reg(H, alpha=1.0, p=3):
    # Compute the sign function (sgn(H))
    sign_H = binarize_ste(H)
    # Compute the difference (H - sgn(H))
    diff = H - sign_H
    # Compute the element-wise absolute value raised to the power of p
    abs_diff_p = torch.abs(diff) ** p
    # Sum over all elements
    norm_p = torch.sum(abs_diff_p)
    # Multiply by alpha
    alpha = alpha * (1 / H.numel())
    result = alpha * norm_p
    return result


class Combined(nn.Module):
    def __init__(self, loss1, loss2, beta):
        super().__init__()
        self.beta = beta
        self.loss1 = loss1
        self.loss2 = loss2

    def forward(self, descriptors, labels, miner_outputs):
        l1 = self.loss1(descriptors, labels, miner_outputs)
        l2 = self.loss2(descriptors, labels, miner_outputs)
        #print("L1: ", l1.item(), "L2: ", l2.item())
        return l1 + self.beta * l2


class GreedyHashLoss_Soft(nn.Module):
    def __init__(self, alpha=1.0, beta=50, base=0.0, reg=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.reg = reg
        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=alpha, beta=beta, base=base, distance=HammingDistanceSoft()
        )
        self.mse = nn.MSELoss()

    def forward(self, descriptors, labels, miner_outputs):
        l1 = self.loss_fn(descriptors, labels, miner_outputs)
        reg_loss = greedy_hash_reg(descriptors, alpha=self.reg)
        return l1 + reg_loss


class GreedyHashLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=50, base=0.0, reg=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.reg = reg
        self.loss_fn = losses.MultiSimilarityLoss(
            alpha=alpha, beta=beta, base=base, distance=HammingDistanceSTE()
        )
        self.mse = nn.MSELoss()

    def forward(self, descriptors, labels, miner_outputs):
        l1 = self.loss_fn(descriptors, labels, miner_outputs)
        reg_loss = greedy_hash_reg(descriptors, alpha=self.reg)
        return l1 + reg_loss


class RegMSE(nn.Module):
    def forward(self, descriptors, labels, miner_outputs):
        descriptors_squared = descriptors**2
        residual = descriptors_squared - 1
        return (residual**2).mean()


class RegSTE(nn.Module):
    def forward(self, descriptors, labels, miner_outputs):
        res = torch.abs(descriptors - torch.sign(descriptors))
        return torch.mean(res)


class RegABS(nn.Module):
    def forward(self, descriptors, labels, miner_outputs):
        loss = torch.mean(torch.abs(torch.abs(descriptors) - 1))
        return loss 
    
class RegHinge(nn.Module): 
    def forward(self, descriptors, labels, miner_outputs, delta=0.1): 
        loss = torch.mean(torch.max(torch.zeros_like(descriptors), delta - torch.abs(descriptors)))
        return loss
    

def get_loss(loss_name):
    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=0.07)
    if loss_name == "CircleLoss":
        return losses.CircleLoss(
            m=0.4, gamma=80
        )  # these are params for image retrieval
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
        )
    if loss_name == "MultiSimilarityLossRegMSE":
        return Combined(
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            ),
            RegMSE(),
            0.15,
        )
    if loss_name == "MultiSimilarityLossRegSTE":
        return Combined(
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            ),
            RegSTE(),
            0.15,
        )
    
    if loss_name == "MultiSimilarityLossRegHinge":
        return Combined(
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            ),
            RegHinge(),
            1.0,
        )
    
    if loss_name == "MultiSimilarityLossRegABS":
        return Combined(
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            ),
            RegABS(),
            1/0.9,
        )
    if loss_name == "HammingSTEMultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=HammingDistanceSTE()
        )
    if loss_name == "HammingSoftMultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=HammingDistanceSoft()
        )
    if loss_name == "CombinedHammingSTEMultiSimilarityLoss":
        return Combined(
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            ),
            losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=HammingDistanceSTE()
            ),
            1.0,
        )
    if loss_name == "GreedyHashLoss_soft":
        return GreedyHashLoss_Soft(alpha=1.0, beta=50, base=0.0, reg=1.0)
    if loss_name == "GreedyHashLoss":
        return GreedyHashLoss(alpha=1.0, beta=50, base=0.0, reg=1.0)
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=0, pos_margin=1, distance=DotProductSimilarity()
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=30)
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(
            temperature=0.07
        )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == "TripletMarginLoss":
        return losses.TripletMarginLoss(
            margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )  # or an int, for example 100
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=0.05,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
        )
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


def get_miner(miner_name, margin=0.1):
    if miner_name == "TripletMarginMiner":
        return miners.TripletMarginMiner(
            margin=margin, type_of_triplets="semihard"
        )  # all, hard, semihard, easy
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == "HammingSTEMultiSimilarityMiner":
        return miners.MultiSimilarityMiner(
            epsilon=margin, distance=HammingDistanceSTE()
        )
    if miner_name == "HammingSoftMultiSimilarityMiner":
        return miners.MultiSimilarityMiner(
            epsilon=margin, distance=HammingDistanceSoft()
        )
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity()
        )
    return None
