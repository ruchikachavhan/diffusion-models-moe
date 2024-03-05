import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import GEGLU
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig

def relu(gate: torch.Tensor) -> torch.Tensor:
    '''
    Relu activation function
    '''
    if gate.device.type != "mps":
        return F.relu(gate)
    # mps: gelu is not implemented for float16
    return F.relu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

def test_relu(model):
    '''
        Test the relu activation function for all blocks
    '''
    for name, module in model.named_modules():
        if isinstance(module, GEGLU) and 'ff.net' in name:
            x = torch.randn(1, 3, 512, 512)
            y = module.gelu(x)
            assert torch.all(y >= 0), f"Relu failed for {name}"
    print("Relu test passed")

def find_and_change_geglu(model, blocks_to_change=['down_block', 'mid_block', 'up_block']):
    num_changed = 0
    # iterate thrugh modules and change the activation function
    for name, module in model.named_modules():
        if isinstance(module, GEGLU) and 'ff.net' in name:
            if any([block in name for block in blocks_to_change]):
                # change the activation function to relu
                module.gelu = relu
                num_changed += 1
    
    test_relu(model)
    print(f'Number of changed modules: {num_changed}')
    return model

