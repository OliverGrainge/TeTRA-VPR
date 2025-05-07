from models.helper import get_model 

import torch.nn as nn
import torch 


def measure_mem(model): 
    sd = model.state_dict()
    mem = sum(p.numel() * p.element_size() for p in sd.values())
    print(f"Model size: {mem / 1024 / 1024} MB")
    return mem

model = get_model(backbone_arch="vitbaset", agg_arch="boq", image_size=322, desc_divider_factor=1)
measure_mem(model)
model.deploy()
measure_mem(model)
