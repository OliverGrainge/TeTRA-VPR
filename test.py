from models.helper import get_model 

import torch.nn as nn
import torch 
model = get_model(backbone_arch="vitbaset", agg_arch="salad", image_size=322)

def _freeze_backbone(model: nn.Module, unfreeze_n_last_layers: int = 1):
    backbone = model.backbone

    for param in backbone.parameters(): 
        param.requires_grad = False

    # set dropout to 0 for all layers except the last unfreeze_n_last_layers
    for block in backbone.transformer.layers[:-unfreeze_n_last_layers]:
        for name, module in block.named_modules(): 
            if isinstance(module, nn.Dropout): 
                module.p = 0.0
                
    # only train the last unfreeze_n_last_layers
    for block in (backbone.transformer.layers[-unfreeze_n_last_layers:]):
        for param in block.parameters(): 
            param.requires_grad = True  
    
    # make sure the backbone is in fully quantized mode 
    for module in model.modules(): 
        if hasattr(module, "set_qfactor"):
            module.set_qfactor(1.0) 

    return model 



def _load_backbone_weights(model: nn.Module, backbone_weights_path: str):
    sd = torch.load(backbone_weights_path, weights_only=False)["state_dict"]

    new_sd = {}
    for key, value in sd.items(): 
        if key.startswith("student"): 
            new_sd[key.replace("student.", "")] = value 

    model.backbone.load_state_dict(new_sd, strict=True)
    return model 


model = _freeze_backbone(model, unfreeze_n_last_layers=1)
model.eval()

_load_backbone_weights(model, backbone_weights_path="checkpoints/TeTRA-pretrain/Student[VitbaseT322]-Teacher[DinoV2]-Aug[Severe]/epoch=2-step=42000-train_loss_step=0.000.ckpt")