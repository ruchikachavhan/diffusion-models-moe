import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
sys.path.append(os.getcwd())
import utils
from diffusers.models.activations import LoRACompatibleLinear, GEGLU
from transformers import CLIPTokenizer, CLIPTextModel
from neuron_receivers import WandaRemoveNeurons, RemoveNeurons, WandaRemoveNeuronsFast, MultiConceptRemoverWanda
import pandas as pd
import argparse
import tqdm
from concept_checkers import BaseConceptChecker, NudityChecker, ArtStyleChecker, art_styles, MemorizedPromptChecker

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
    'Van Gogh': 0.05,
    'Monet': 0.05,
    'Pablo Picasso': 0.05,
    'naked': 0.01,
    'memorize': 0.01, 
}

        

def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', type=bool, default=False, help='debug')
    args.add_argument('--gpu', type=int, default=0, help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='n layers')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', default='art,naked,memorize', help='List of concepts to remove')
    args.add_argument('--dataset_path', default='modularity/datasets/combined_prompts.csv', help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')

    args = args.parse_args()
    return args

def main():
    args = args_parser()
    # Step 1 - Read the prompts file
    if args.dataset_path.endswith('.csv'):
        test_prompts = pd.read_csv(args.dataset_path)
        # remove head of the dataframe
        test_prompts = test_prompts[1:]
    elif args.dataset_path.endswith('.parquet'):
        test_prompts = pd.read_parquet(args.dataset_path)
        # replace the column name "caption" with "prompt"
        test_prompts = test_prompts.rename(columns={'caption': 'prompt'})
        # add another column of name "concept" withall values 'memorize'
        test_prompts['concept'] = ['memorize'] * len(test_prompts)

    # Get concept checkets
    memorized_checker = MemorizedPromptChecker(device=args.gpu, objects=things, neg_objects=None)
    nudity_checker = NudityChecker(device=args.gpu, objects=humans, neg_objects=things)
    art_style_checker = ArtStyleChecker(device=args.gpu, objects=things, neg_objects=humans)

    # initialise neuron receivers
    all_concepts_to_remove = []
    if 'naked' in args.concepts_to_remove:
        all_concepts_to_remove += ['naked']
    if 'art' in args.concepts_to_remove:
        all_concepts_to_remove += art_styles
    if 'memorize' in args.concepts_to_remove:
        all_concepts_to_remove += ['memorize']
    
    # SD model
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    model = model.to(args.gpu)
    
    neuron_remover = MultiConceptRemoverWanda(root = args.root_template, seed = args.seed, 
            T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, 
            keep_nsfw = args.keep_nsfw, concepts_to_remove = all_concepts_to_remove, wanda_thr=wanda_thr)

    output_path = os.path.join('benchmarking results', 'unified', 'memorized' if 'memorize' in args.concepts_to_remove else '')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = {}

    for index, row in tqdm.tqdm(test_prompts.iterrows()):
        prompt = row['prompt'].split("\n")[0]
        labels = row['concept']
        remove_nudity = nudity_checker.decide(prompt)
        remove_art_style = art_style_checker.decide(prompt)
        remove_memorized = memorized_checker.decide(prompt)
        pred = [remove_nudity, remove_art_style]
        concept_remove = []
        if remove_nudity == 'naked' and 'naked' in all_concepts_to_remove:
            concept_remove.append('naked')
        if remove_art_style != 'none' and remove_art_style in all_concepts_to_remove:
            concept_remove.append(remove_art_style)
        if remove_memorized == 'memorize' and 'memorize' in all_concepts_to_remove:
            concept_remove.append('memorize')

        print(f'Prompt: {prompt}', f'Prediction: {concept_remove}')
        results[prompt] = {}
        results[prompt]['labels'] = labels
        results[prompt]['pred'] = concept_remove
        # check if concepts need to be removed
        output, single_image_removal = neuron_remover.remove_concepts(model, prompt, concept_remove)

        p = prompt[:20]
        output.save(output_path + f'/{index}.png')
        if single_image_removal is not None:
            for i, im in enumerate(single_image_removal):
                im.save(output_path + f'/{index}_{i}.png')
        
    # save results to file
    with open(output_path + '/results.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()