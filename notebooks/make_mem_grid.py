import torch
import numpy as np
from diffusers import UNet2DConditionModel, StableDiffusionPipeline



seed_range = 5
exp_range = 10

prompt = "Prince Reunites With Warner Brothers, Plans New Album" 

image_grid = {}
for exp in range(0,exp_range):
    image_grid[exp] = {}
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5',  subfolder="unet", torch_dtype=torch.float16)
    path = 'eval_checkpoints_ap/memorize_%s_0.3.pt' % str(exp)
    unet.load_state_dict(torch.load(path))
    model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, unet=unet)
    model = model.to('cuda:6')
    for seed in range(seed_range):
        print("Seed: ", seed)
        # generate image
        torch.manual_seed(seed)
        np.random.seed(seed)
        out = model(prompt).images[0]
        out = out.resize((100, 100))
        image_grid[exp%10][seed] = out

# plot the images in a grid
import matplotlib.pyplot as plt
import numpy as np
import os

fig, axs = plt.subplots(exp_range, seed_range, figsize=(seed_range, exp_range))
for i in range(0,exp_range):
    for j in range(seed_range):
        axs[i, j].imshow(image_grid[i][j])
        # label axis of first image in each row
        if j == 0:
            # print on top 
            axs[i, j].set_ylabel(f'Exp {i}', rotation=90, fontsize=5)
        # label axis of first image in each column
        if i == 0:
            axs[i, j].set_title(f'Seed {j}', fontsize=5)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
# plt.tight_layout()
plt.savefig('mem_grid_ap_ann.png')


    


