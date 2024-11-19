import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from PIL import Image 


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def get_feature_dim(model, transform):
    x = torch.randint(0, 255, size=(3, 512, 512), dtype=torch.uint8)
    x_np = x.numpy()
    x_img = Image.fromarray(x_np.transpose(1, 2, 0))
    x_transformed = transform(x_img)
    print(x_transformed.shape)
    features = model(x_transformed[None, :].to(next(model.parameters()).device))
    return features["global_desc"].shape[1]



def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model