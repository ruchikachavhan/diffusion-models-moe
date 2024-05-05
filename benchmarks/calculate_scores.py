import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
from torchvision import transforms
sys.path.append(os.getcwd())
import utils
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import pandas as pd
import argparse
import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

def calculate_fid(original_images, generated_images):
    fid = FID()
    fid.update(original_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()


things = ['cat', 'dog', 'bird', 'car', 'bear', 'bat', 'ball', 'apple', 'banana', 'house', 'tree', 'flower', 'room', 'building', 'wall', 'chicken', 'duck']
humans = ['man', 'woman', 'child', 'boy', 'girl', 'kid', 'teacher', 'nurse', 'doctor', 'professor', 'guy', 'couple']
wanda_thr = {
    'Alex Alemany,painter': 0.05,
    'John Howe,illustrator': 0.05,
    'Antonio J. Manzanedo': 0.05,
    'Alpo Jaakola,painter': 0.05,
    'Abraham Mintchine,painter': 0.05,
    'Apollinary Vasnetsov,landscape art,history painting': 0.05,
    'John Constable,realism,landscape art': 0.05,
    'Johannes Vermeer,genre painting,portrait': 0.05,
    'Nicolas Mignard,mythological painting,painter': 0.05,
    'John Whitcomb': 0.05,
    'Amedeo Modigliani,landscape art,portrait': 0.05,
    'Jordan Grimmer': 0.05,
    'A.J.Casson': 0.05,
    'Akira Toriyama,mangaka,fantasy': 0.05,
    'Salvador Dali': 0.05,
    'Greg Rutkowski': 0.05,
    'Jeremy Mann': 0.05,
    'Van Gogh': 0.02,
    'Monet': 0.05,
    'Pablo Picasso': 0.05,
    'naked': 0.01,
    'memorize': 0.01, 
    '5artists_naked': '',
    '5artists': 0.95,
    '10artists': 0.95,
    '50artists': 0.95,
}
uce_models_dict = {
    '50artists': 'erased-50artists-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '1artist': 'erased-algernon blackwood-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '10artists': 'erased-asger jorn_eric fischl_johannes vermeer_apollinary vasnetsov_naoki urasawa_nicolas mignard_john whitcomb_john constable_warwick globe_albert marquet-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '5artists': 'erased-juliana huxtable_valerie hegarty_wendy froud_kobayashi kiyochika_paul laffoley-towards_art-preserve_true-sd_1_4-method_replace.pt',
}
    
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


def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', action='store_true', help='debug')
    args.add_argument('--gpu', type=int, default=7, help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='n layers')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', default=None, help='List of concepts to remove')
    args.add_argument('--dataset_type', default=None, help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=4, type=int, help='Batch size')

    args = args.parse_args()
    return args

def main():
    args = args_parser()
    # Step 1 - Read thhe COCO dataset

    if args.dataset_type == 'coco':
        imgs, anns = utils.coco_dataset('../COCO-vqa', 'val', 30000)
        print("Evaluating on COCO dataset", len(imgs), len(anns))
        num_images = len(imgs)
        # make a dataloadet for coco
    elif args.dataset_type == 'holdout':
        print("Testing model on Holdout dataset of artists")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # read csv file with holdout artists
        data = pd.read_csv('modularity/datasets/holdout100_prompts.csv')
        # create a list of artsist from data
        prompts = data['prompt'].tolist()
        num_images = len(prompts)
    else:
        raise ValueError("Dataset type not found")

    if args.fine_tuned_unet is None:
        output_path = f'benchmarking results/unified/{args.dataset_type}/{args.concepts_to_remove}'
    else:
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'
    
    print(f"Output path: {output_path}")
    lpips = LPIPS(net_type='squeeze')
    results = {}
    results['lpips'] = {}
    results['lpips']['SD'] = []
    results['lpips']['After_removal'] = []

    for iter in range(num_images):
        if args.dataset_type == 'coco':
            orig_image = image_loader(imgs[iter])
            gen_image = image_loader(os.path.join(output_path, f'sd_{iter}.png'))
            removal_image = image_loader(os.path.join(output_path, f'removed_{iter}.png'))
            score_sd = lpips(orig_image, gen_image)
            score_after_removal = lpips(orig_image, removal_image)
        else:
            gen_image = image_loader(os.path.join(output_path, f'sd_{iter}.png'))
            removal_image = image_loader(os.path.join(output_path, f'removed_{iter}.png'))

            score_sd = torch.tensor(0.0)
            score_after_removal = lpips(gen_image, removal_image)

        print(f"Image {iter} - LPIPS score before removal: {score_sd.item()} - LPIPS score after removal: {score_after_removal.item()}")

        results['lpips']['SD'].append(score_sd.item())
        results['lpips']['After_removal'].append(score_after_removal.item())

    # calcualate LPIPS score mean and std
    mean = np.mean(results['lpips']['SD'])
    std = np.std(results['lpips']['SD'])
    results['lpips']['SD'] = (mean, std)
    mean = np.mean(results['lpips']['After_removal'])
    std = np.std(results['lpips']['After_removal'])
    results['lpips']['After_removal'] = (mean, std)

    print(results)
    # save 
    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    
    
    

if __name__ == '__main__':
    main()