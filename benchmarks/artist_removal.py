import torch
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
import tqdm
from diffusers.models.activations import GEGLU, GELU
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from diffusers.pipelines.stable_diffusion import safety_checker
from transformers import CLIPProcessor, CLIPModel

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker


best_ckpt_dict = {
    'Van Gogh': 'Van Gogh_0.0.pt',
    'Monet': 'Monet_0.0.pt',
    'Pablo Picasso': 'Pablo Picasso_0.0.pt',
    'Salvador Dali': 'Salvador Dali_0.0.pt',
    'Leonardo Da Vinci': 'Leonardo Da Vinci_0.0.pt',
    'all_imagenette_objects': 'all_imagenette_objects.pt',
}

uce_models_dict = {
    '50artists': 'erased-50artists-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '1artist': 'erased-algernon blackwood-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '10artists': 'erased-asger jorn_eric fischl_johannes vermeer_apollinary vasnetsov_naoki urasawa_nicolas mignard_john whitcomb_john constable_warwick globe_albert marquet-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '5artists': 'erased-juliana huxtable_valerie hegarty_wendy froud_kobayashi kiyochika_paul laffoley-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '100artists': 'erased-100artists-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Van Gogh': 'erased-van gogh-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Monet': 'erased-claude monet-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Pablo Picasso': 'erased-pablo picasso-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Salvador Dali': 'erased-salvador dali-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Leonardo Da Vinci': 'erased-leonardo da vinci-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Rembrandt': 'erased-rembrandt-towards_art-preserve_true-sd_1_4-method_replace.pt',
}

concept_ablation_dict = {
    'Van Gogh': 'vangogh',
    'Monet': 'monet',
    'Pablo Picasso': 'picasso',
    'Salvador Dali': 'salvador_dali',
    'Leonardo Da Vinci': 'davinci',
    'Rembrandt': 'rembrandt',
}

class HoldoutDataset(torch.utils.data.Dataset):
    def __init__(self, prompts):
        self.prompts = prompts['prompt'].tolist()
        self.seeds = prompts['evaluation_seed'].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        seed = self.seeds[idx]
        return prompt,  seed
    

# python benchmarks/artist_removal.py --concepts_to_remove 'Van Gogh' --fine_tuned_unet 'union-timesteps' --gpu 0
def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', action='store_true', help='debug')
    args.add_argument('--gpu', type=int, default=0, help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='n layers')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', default=None, help='List of concepts to remove')
    args.add_argument('--dataset_type', default='artist_painting', help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/art/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=1, type=int, help='Batch size')

    args = args.parse_args()
    return args

def main():
    args = args_parser()

    data = pd.read_csv(f'modularity/datasets/concept_removal_{args.concepts_to_remove}.csv')
    dataloader = torch.utils.data.DataLoader(HoldoutDataset(data), batch_size=args.batch_size, shuffle=False)

    # Pre-trained model
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to(args.gpu)

    if args.fine_tuned_unet == 'uce':
        # load a baseline model and fine tune it
        unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(os.path.join('../unified-concept-editing/models', uce_models_dict[args.concepts_to_remove])))
        remover_model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet, torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'

    if args.fine_tuned_unet == 'concept-ablation':
        remover_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        model_path = os.path.join('../concept-ablation/diffusers', 'logs_ablation', concept_ablation_dict[args.concepts_to_remove], 'delta.bin')
        print(f"Loading model from {model_path}")
        model_ckpt = torch.load(model_path)
        if 'text_encoder' in model_ckpt:
            remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
        for name, params in remover_model.unet.named_parameters():
            if name in model_ckpt['unet']:
                params.data.copy_(model_ckpt['unet'][f'{name}'])
        # remover_model.load_model(os.path.join('../concept-ablation/diffusers', 'logs_ablation', args.concepts_to_remove, 'delta.bin'))
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'
    
    if args.fine_tuned_unet == 'union-timesteps':
        unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", torch_dtype=torch.float16)
        best_ckpt_path = os.path.join('results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/', 'checkpoints', f'{best_ckpt_dict[args.concepts_to_remove]}')
        print(f"Loading model from {best_ckpt_path}")
        unet.load_state_dict(torch.load(best_ckpt_path))
        remover_model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', unet=unet, torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'

    if args.fine_tuned_unet is None:
        # initalise Wanda neuron remover
        path_expert_indx = os.path.join(args.root_template % (str(args.seed), args.concepts_to_remove), 'skilled_neuron_wanda', str(wanda_thr[args.concepts_to_remove]))
        print(f"Path expert index: {path_expert_indx}")
        neuron_remover = WandaRemoveNeuronsFast(seed = args.seed, path_expert_indx = path_expert_indx, T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, keep_nsfw =True)
        output_path = f'benchmarking results/unified/{args.dataset_type}/{args.concepts_to_remove}'


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # test model on dataloader
    for iter, (prompt, seed) in enumerate(dataloader):

        # check if image is present in putput path
        if os.path.exists(os.path.join(output_path, f"sd_{iter * args.batch_size}.png")):
            print(f"Skipping iteration {iter}")
            continue

        print("Iteration number", iter, prompt, seed)
        prompt = [p for p in prompt]

        torch.manual_seed(seed[0])
        np.random.seed(seed[0])
        # remove neurons
        orig_image = model(prompt, safety_checker=safety_checker_).images[0]
        orig_image.save(os.path.join(output_path, f"sd_{iter}.png"))

        if args.concepts_to_remove is not None:
            if args.fine_tuned_unet in ['uce', 'concept-ablation', 'union-timesteps']:
                torch.manual_seed(seed[0])
                np.random.seed(seed[0])
                removal_images = remover_model(prompt=prompt, safety_checker=safety_checker_).images[0]
            elif args.fine_tuned_unet is None:
                neuron_remover.reset_time_layer()
                removal_images, _ = neuron_remover.observe_activation(model, prompt)

            removal_images.save(os.path.join(output_path, f"removed_{iter}.png"))

    # Calculate CLIP score between text and image
                
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    average_clipscore = 0
    average_acc = 0
    for iter, (prompt, seed) in enumerate(dataloader):
        prompt = [p for p in prompt]
        orig_image = Image.open(os.path.join(output_path, f"sd_{iter}.png"))
        removal_image = Image.open(os.path.join(output_path, f"removed_{iter}.png"))

        # encode the text
        text_inputs = clip_processor(prompt, return_tensors="pt", padding=True)
        text_features = clip_model.get_text_features(**text_inputs)

        # encode the images
        image_inputs = clip_processor(images=orig_image, return_tensors="pt")
        orig_image_features = clip_model.get_image_features(**image_inputs)

        image_inputs = clip_processor(images=removal_image, return_tensors="pt")
        removal_image_features = clip_model.get_image_features(**image_inputs)

        # calculate the similarity
        similarity_orig = torch.nn.functional.cosine_similarity(text_features, orig_image_features)
        similarity_removed = torch.nn.functional.cosine_similarity(text_features, removal_image_features)

        image_sim = torch.nn.functional.cosine_similarity(orig_image_features, removal_image_features)

        average_acc += 1 if similarity_orig > similarity_removed else 0

        average_clipscore += image_sim.item()

    average_clipscore /= len(dataloader)
    average_acc /= len(dataloader)

    print(f"Average CLIP score: {average_clipscore}")
    print(f"Average accuracy: {average_acc}")

    # save results in text file
    with open(os.path.join(output_path, 'results.txt'), 'w') as f:
        f.write(f"Average CLIP score: {average_clipscore}\n")
        f.write(f"Average accuracy: {average_acc}\n")



if __name__ == '__main__':
    main()
