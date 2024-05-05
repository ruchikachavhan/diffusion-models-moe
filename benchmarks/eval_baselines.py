import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
from torchvision import transforms
sys.path.append(os.getcwd())
import utils
from diffusers.models.activations import LoRACompatibleLinear, GEGLU
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from neuron_receivers import WandaRemoveNeurons, RemoveNeurons, WandaRemoveNeuronsFast, MultiConceptRemoverWanda
import pandas as pd
import argparse
import tqdm
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from concept_checkers import BaseConceptChecker, NudityChecker, ArtStyleChecker, art_styles, MemorizedPromptChecker
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

def calculate_fid(original_images, generated_images):
    fid = FID()
    fid.update(original_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()


def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', action='store_true', help='debug')
    args.add_argument('--gpu', type=int, default=0, help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='n layers')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', required=True, help='List of concepts to remove')
    args.add_argument('--dataset_path', default='modularity/datasets/holdout100_prompts.csv', help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s/skilled_neuron_wanda', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')

    args = args.parse_args()
    return args


# Copied from https://github.com/rohitgandikota/unified-concept-editing/blob/main/eval-scripts/lpips_eval.py
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)
def main():

    args = args_parser()

    sd_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    sd_model = sd_model.to(args.gpu)
    baseline_name = 'uce'
    if baseline_name == 'uce':
        if 'art' in args.concepts_to_remove:
            # read their checkpoint
            path = f'/raid/s2265822/unified-concept-editing/models/erased-{args.concepts_to_remove}-towards_art-preserve_true-sd_1_4-method_replace.pt'
            unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.float16)
            unet.load_state_dict(torch.load(path))
            model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet, torch_dtype=torch.float16)
            model = model.to(args.gpu)

            save_dir = f'/raid/s2265822/unified-concept-editing/results/erased-{args.concepts_to_remove}-towards_art-preserve_true-sd_1_4-method_replace'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
          
    elif baseline_name == 'ablation':
        path = f'/raid/s2265822/'

    # Load the dataset of csv   
    prompts = pd.read_csv(args.dataset_path)
    lpips = LPIPS(net_type='alex')

    print(prompts.keys())

    scores = []
    for index, row in prompts.iterrows():
        prompt = row['prompt']

        if not os.path.exists(os.path.join(save_dir, f'base_image_{index}.png')):
            # process the prompt 
            base_image = sd_model(prompt).images[0]

            gen_image = model(prompt).images[0]

            # save images
            base_image.save(f'{save_dir}/base_image_{index}.png')
            gen_image.save(f'{save_dir}/gen_image_{index}.png')

        base_image = image_loader(f'{save_dir}/base_image_{index}.png')
        gen_image = image_loader(f'{save_dir}/gen_image_{index}.png')

        # calculate LPIPS between the two images
        score = lpips(base_image, gen_image)
        print("LPIPS Score: ", score)

        scores.append(score.item())

    # save the scores
        
    mean = np.mean(scores)
    results = {}
    results['lpips'] = mean
    results['std'] = np.std(scores)

    with open(f'{save_dir}/lpips_scores.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()