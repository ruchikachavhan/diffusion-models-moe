import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
from torchvision import transforms
sys.path.append(os.getcwd())
import utils
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from neuron_receivers import WandaRemoveNeuronsFast
import pandas as pd
import argparse
import open_clip
import glob
from tqdm import tqdm
import logging
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional
from diffusers.models.activations import GEGLU, GELU
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from diffusers.pipelines.stable_diffusion import safety_checker

def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    print(f"Reading JSON lines from {filename}")
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm(
            fp.readlines(), desc=f"Reading JSON lines from {filename}", unit="lines"
        ):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex
            
def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))

### credit: https://github.com/somepago/DCR
def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.mm(feat_1, feat_2.T)


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
    
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
    args.add_argument('--concepts_to_remove', default='memorize', help='List of concepts to remove')
    args.add_argument('--dataset_type', default='memorize', help='dataset path')
    args.add_argument('--skill_method', default='wanda', help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=4, type=int, help='Batch size')
    args.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")

    args = args.parse_args()
    return args

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

def main():
    args = args_parser()
    # Step 1 - Read thhe Memorization dataset
    dataset_path = f'../diffusion_memorization/sdv1_500_mem_groundtruth'
    dataset = load_jsonlines(f"{dataset_path}/sdv1_500_mem_groundtruth.jsonl")
    print(f"Dataset: {dataset}")

    args.reference_model = 'ViT-g-14'

    
    # Pre-trained model
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to(args.gpu)

    sim_model = torch.jit.load("../diffusion_memorization/sscd_disc_large.torchscript.pt").to(args.gpu)
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=args.gpu,
        )
    ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
    
    if args.fine_tuned_unet == 'concept-ablation':
        remover_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        model_path = os.path.join('../concept-ablation/diffusers', 'logs_ablation', args.concepts_to_remove, 'delta.bin')
        model_ckpt = torch.load(model_path)
        if 'text_encoder' in model_ckpt:
            remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
        for name, params in remover_model.unet.named_parameters():
            if name in model_ckpt['unet']:
                params.data.copy_(model_ckpt['unet'][f'{name}'])
        # remover_model.load_model(os.path.join('../concept-ablation/diffusers', 'logs_ablation', args.concepts_to_remove, 'delta.bin'))
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'

    if args.fine_tuned_unet == 'union-timesteps':
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
        if args.skill_method == 'wanda':
            root_template = f'eval_checkpoints'
            best_ckpt_path = os.path.join(root_template, f'{args.concepts_to_remove}_0.4.pt')
        elif args.skill_method == 'AP':
            root_template = f'eval_checkpoints_ap'
            best_ckpt_path = os.path.join(root_template, f'{args.concepts_to_remove}.pt')
        print(f"Best checkpoint path: {best_ckpt_path}")
        unet.load_state_dict(torch.load(best_ckpt_path))
        remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}/{args.skill_method}' 
        remover_model = remover_model.to(args.gpu)

    if args.fine_tuned_unet is None:
        # initalise Wanda neuron remover
        path_expert_indx = os.path.join(args.root_template % (str(args.seed), args.concepts_to_remove), 'skilled_neuron_wanda', '0.01')
        print(f"Path expert index: {path_expert_indx}")
        neuron_remover = WandaRemoveNeuronsFast(seed = args.seed, path_expert_indx = path_expert_indx, T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, keep_nsfw =True)
        output_path = f'benchmarking results/unified/{args.dataset_type}/{args.concepts_to_remove}/random_seed'


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gen_images, gt_images = [], []
    results = {}
    iter = 0
    for data in dataset:
        prompt = data['caption']
        im_index = data['index']
        print(f"Prompt: {prompt}")
        iter +=1 


        # Step 2 - Generate the image
        if True:
            if args.fine_tuned_unet is None:
                seed = torch.randint(0, 250, (1,)).item()
                neuron_remover.seed = seed
                neuron_remover.reset_time_layer()
                image, _ = neuron_remover.observe_activation(model, prompt)
                # save the image 
                image.save(f"{output_path}/{im_index}.png")
            elif args.fine_tuned_unet in ['concept-ablation', 'union-timesteps']:
                # fix seed
                seed = torch.randint(0, 10000, (1,)).item()
                torch.manual_seed(seed)
                np.random.seed(seed)
                image = remover_model(prompt, safety_checker = safety_checker_).images[0]
                image.save(f"{output_path}/{im_index}.png")
        else:
            print(f"Image already exists at {output_path}/{im_index}.png")
            image = Image.open(f"{output_path}/{im_index}.png")
        
        gen_images = [image]
        # read ground truth images
        gt_folder = f"{dataset_path}/gt_images/{im_index}"
        # read all images in gt_image folder
        gt_images = [Image.open(filename) for filename in glob.glob(f"{gt_folder}/*.png")]
        print(f"Ground truth images: {gt_images}")
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, args.gpu)
        print(f"SSCD similarity: {SSCD_sim}")


        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        SSCD_sim = SSCD_sim.max(0).values
        sims = measure_CLIP_similarity(
                [gt_image] + gen_images,
                prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                args.gpu,
            )
        gt_clip_score = sims[0:1].mean().item()
        gen_clip_score = sims[1:].mean().item()

        results[im_index] = {
            'prompt': prompt,
            'SSCD_sim': SSCD_sim.item(),
            'gt_clip_score': gt_clip_score,
            'gen_clip_score': gen_clip_score,
        }


        print(f"GT clip score: {gt_clip_score}", f"Gen clip score: {gen_clip_score}")
        
    # save the results
    avg_sscd = 0
    avg_gt_clip_score = 0
    avg_gen_clip_score = 0
    for key, value in results.items():
        avg_sscd += value['SSCD_sim']
        avg_gt_clip_score += value['gt_clip_score']
        avg_gen_clip_score += value['gen_clip_score']
    avg_sscd /= len(results)
    avg_gt_clip_score /= len(results)
    avg_gen_clip_score /= len(results)

    results['average'] = {
        'avg_sscd': avg_sscd,
        'avg_gt_clip_score': avg_gt_clip_score,
        'avg_gen_clip_score': avg_gen_clip_score
    }

    with open(f"{output_path}/results.json", 'w') as f:
        json.dump(results, f)

        


if __name__ == '__main__':
    main()