import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import yaml

from models.helper import get_model

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


model = get_model(
    (224, 224),
    "ternary_vit_base",
    "cls",
    config["Model"],
    normalize_output=True,
)


def weight_quant(w):
    scale = 1.0 / w.abs().max().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


sd = torch.load(
    "/home/oliver/Downloads/ternary_vit_base_cls/version_8714/checkpoints/imagenet/ternary_vit_base_cls_epoch(85)_step(66220)_A1[74.1040]_A5[91.4000].ckpt"
)
sd = sd["state_dict"]
new_sd = {}
for key, value in sd.items():
    if key != "fc.weight" and key != "fc.bias":
        new_sd[key.replace("model.", "")] = value
model.load_state_dict(new_sd, strict=False)

print(model.backbone.transformer.layers[8][0].to_qkv.weight.data)
w = model.backbone.transformer.layers[8][1].net[1].weight.data
rmsnorm = model.backbone.transformer.layers[8][1].net[1].rmsnorm
w = model.backbone.transformer.layers[8][0].to_qkv.weight.data
rmsnorm = model.backbone.transformer.layers[8][0].to_qkv.rmsnorm

import matplotlib.pyplot as plt

# Assuming w and rmsnorm are already defined
wr = rmsnorm(w)

# Calculate the mean of absolute values
w_abs_mean = w.abs().mean().item()
wr_abs_mean = wr.abs().mean().item()


resq = (w - weight_quant(w)).abs()
resqrms = (rmsnorm(w) - weight_quant(rmsnorm(w))).abs()

plt.hist(resq.detach().cpu().flatten(), bins=100, alpha=0.5)
plt.hist(resqrms.detach().cpu().flatten(), bins=100, alpha=0.5)
plt.show()
#
print("res")
print(resq.sum())
print(resqrms.sum())
# Print the means
print(w_abs_mean)
print(wr_abs_mean)

# Plot the histograms
plt.hist(wr.detach().cpu().flatten(), bins=100, label="after norm", alpha=0.5)
w_abs_mean = 3 / 2 * wr.detach().cpu().abs().mean()
# Add vertical lines for the means
plt.axvline(
    w_abs_mean,
    color="blue",
    linestyle="--",
    label=f"wr.abs().mean() = {w_abs_mean:.4f}",
)

# Add legend and display the plot
plt.legend()
plt.show()
