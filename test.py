from models.helper import get_model 
import torch 

model = get_model(backbone_arch="vitbaset", agg_arch="boq", image_size=(322,322))

sd = torch.load("checkpoints/TeTRA-finetune/logistic/VitbaseT322_BoQ-DescDividerFactor[1]/epoch=1-MSLS_val_q_R1=84.71.ckpt", weights_only=True, map_location="cpu")["state_dict"]
                                                                                                                                                                
print(list(model.state_dict().keys()))
print("===============================" * 100)
print(list(sd.keys()))
print("===============================" * 100)
model.load_state_dict(sd, strict=True)