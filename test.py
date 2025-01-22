from models.helper import get_model 
import torch 

model = get_model(preset="DinoV2_BoQ")
vit = get_model(backbone_arch="vitbaset", agg_arch="gem", image_size=[322, 322])

print(model)

backbone = model.backbone 

img = torch.randn(1, 3, 322, 322)


out = backbone.forward_cls(img) 
out2 = backbone.forward_cls(img)

print(out.shape, out2.shape)