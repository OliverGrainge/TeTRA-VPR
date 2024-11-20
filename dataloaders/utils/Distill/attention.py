import torch
from einops import rearrange


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_attn(model):
    attention_matrices = []
    hooks = []

    def dinov2_hook_fn(module, input, output):
        B, N, C = input[0].shape
        qkv = (
            module.qkv(input[0])
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * module.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attention_matrices.append(attn)

    def vit_hook_fn(module, input, output):
        qkv = module.to_qkv(input[0]).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=module.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * module.scale
        attn = dots.softmax(dim=-1)
        attention_matrices.append(attn)

    for name, module in model.named_modules():
        if hasattr(module, "qkv"):
            # This is likely a DINOv2 attention module
            hook = module.register_forward_hook(dinov2_hook_fn)
            hooks.append(hook)
        elif hasattr(module, "to_qkv"):
            # This is likely a ViT attention module
            hook = module.register_forward_hook(vit_hook_fn)
            hooks.append(hook)

    return attention_matrices, hooks
