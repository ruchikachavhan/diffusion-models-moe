import torch
from torch import nn
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline
from diffusers.models.activations import LoRACompatibleLinear

model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)

total_ffn_params = 0
total_attn_params = 0
for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name:
            total_ffn_params += sum(p.numel() for p in module.parameters())
        elif 'attn' in name:
            total_attn_params += sum(p.numel() for p in module.parameters())

print(f"Total number of parameters in the feedforward networks: {total_ffn_params}")
# total number of paraneters in unet
total_unet_params = sum(p.numel() for p in model.unet.parameters())

print(f"Total number of parameters in the UNet: {total_unet_params}")
print("Percentage of parameters in the feedforward networks: ", total_ffn_params / (total_ffn_params + total_unet_params) * 100, "%")

print(f"Total number of parameters in the attention modules: {total_attn_params}")
print("Percentage of parameters in the attention modules: ", total_attn_params / (total_attn_params + total_unet_params) * 100, "%")

print(model.unet)