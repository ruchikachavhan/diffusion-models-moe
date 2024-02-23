import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
sys.path.append(os.getcwd())
import utils as dm_utils
import eval_coco as ec
from fvcore.nn import FlopCountAnalysis
from transformers import CLIPTextModel, CLIPTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../COCO-vqa', help='path to the coco dataset')
    parser.add_argument('--blocks-to-change', nargs='+', default=['down_block', 'mid_block', 'up_block'], help='blocks to change the activation function')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--res-path', type=str, default='results/stable-diffusion/', help='path to store the results of moefication')
    parser.add_argument('--dbg', action='store_true', help='debug mode')
    parser.add_argument('--num-images', type=int, default=1000, help='number of images to test')
    parser.add_argument('--fine-tuned-unet', type = str, default = None, help = "path to fine-tuned unet model")
    parser.add_argument('--model-id', type=str, default="runwayml/stable-diffusion-v1-5", help='model id')
    parser.add_argument('--timesteps', type=int, default=51, help='number of denoising time steps')
    parser.add_argument('--num-layer', type=int, default=3, help='number of layers')
    parser.add_argument('--topk-experts', type=float, default=1, help='ratio of experts to select')
    parser.add_argument('--templates', type=str, 
                        default='{}.{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight',
                        help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

    args = parser.parse_args()
    return args

# for step, batch in enumerate(train_dataloader):
#             with accelerator.accumulate(unet):
#                 # Convert images to latent space
#                 latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
#                 latents = latents * vae.config.scaling_factor

#                 # Sample noise that we'll add to the latents
#                 noise = torch.randn_like(latents)
#                 if args.noise_offset:
#                     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
#                     noise += args.noise_offset * torch.randn(
#                         (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
#                     )
#                 if args.input_perturbation:
#                     new_noise = noise + args.input_perturbation * torch.randn_like(noise)
#                 bsz = latents.shape[0]
#                 # Sample a random timestep for each image
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
#                 timesteps = timesteps.long()

#                 # Add noise to the latents according to the noise magnitude at each timestep
#                 # (this is the forward diffusion process)
#                 if args.input_perturbation:
#                     noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
#                 else:
#                     noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

#                 # Get the text embedding for conditioning
#                 encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

#                 # Get the target for loss depending on the prediction type
#                 if args.prediction_type is not None:
#                     # set prediction_type of scheduler if defined
#                     noise_scheduler.register_to_config(prediction_type=args.prediction_type)

#                 if noise_scheduler.config.prediction_type == "epsilon":
#                     target = noise
#                 elif noise_scheduler.config.prediction_type == "v_prediction":
#                     target = noise_scheduler.get_velocity(latents, noise, timesteps)
#                 else:
#                     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

#                 # Predict the noise residual and compute loss
#                 model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]


def main():
    args = get_args()
    dm_utils.make_dirs(args)

    model, num_geglu = dm_utils.get_sd_model(args)
    model = model.to(args.gpu)

    text = "A person is riding a horse"
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # text = tokens.encode(text, return_dict=False)[0]
    tokens = tokenizer(text, return_tensors="pt")
    # put the text to the gpu
    tokens = {k: v.to(args.gpu) for k, v in tokens.items()}
    print(text)
    print(tokens)

    text_feats = model.text_encoder(tokens['input_ids'], return_dict=False)[0]
    print(text_feats.shape)

    timesteps = torch.randint(0, 51, (1,), device=args.gpu).to(torch.long)
    flops =  FlopCountAnalysis(model.unet, (torch.rand(1,4,64,64).to(args.gpu).to(torch.float16), timesteps, text_feats.to(args.gpu).to(torch.float16)))
    print(flops.total(), flops.by_operator())

if __name__ == '__main__':
    main()