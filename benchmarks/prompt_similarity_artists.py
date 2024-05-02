import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
sys.path.append(os.getcwd())
import utils
from diffusers.models.activations import LoRACompatibleLinear, GEGLU
from neuron_receivers import WandaRemoveNeurons, RemoveNeurons, WandaRemoveNeuronsFast
import pandas as pd
import argparse
import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 
from diffusers.pipelines.stable_diffusion import safety_checker
sys.path.append('benchmarks')
from benchmark_utils import remove_neuron, save_image_embeddings, ConceptChecker, preprocess_concepts, concept_types, global_concept, ConceptCheckerSingle

def sc(self, clip_input, images):
    return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', type=bool, default=False, help='debug')
    args.add_argument('--gpu', type=str, default='cuda', help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='timesteps')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--cos_thr', default=0.8, type=float, help='Threshold for CLIP similarity')
    args.add_argument('--timesteps', default=51, type=int, help='Threshold for Bleu scre')

    args = args.parse_args()
    return args
global_concept = {
    'artist': ['painting', 'art'],
    'naked': ['naked', 'sexy', 'sensual', 'hot', 'in lingerie'],
}
def main():
    args = args_parser()
    # Step 1 - Read the prompts file
    prompts = pd.read_csv('modularity/datasets/combined_prompts.csv')
    # remove head of the dataframe
    prompts = prompts[1:]
    
    # Step 2 - Load the pretrained model
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    model = model.to(args.gpu)

    num_modules = 0
    for name, module in model.unet.named_modules():
        if isinstance(module, LoRACompatibleLinear) and 'ff.net' in name and not 'proj' in name:
            num_modules += 1
    print("Number of modules: ", num_modules)

    # Step 3 - CLIp text encoder
    # This is a temporary step and will be removed once concepts to remove are finalised
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(args.gpu)

    # get anchor image embeddings
    root = f'results/results_seed_{args.seed}/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s/images'
    
    # concepts_to_remove = open('modularity/datasets/famous_artists.txt').read().split('\n')
    concepts_to_remove = {'Alex Alemany,painter': 'artist',
                    'John Howe,illustrator': 'artist',
                    'Antonio J. Manzanedo': 'artist',
                    'Alpo Jaakola,painter': 'artist',
                    'Abraham Mintchine,painter': 'artist',
                    'Apollinary Vasnetsov,landscape art,history painting': 'artist',
                    'John Constable,realism,landscape art': 'artist',
                    'Johannes Vermeer,genre painting,portrait': 'artist',
                    'Nicolas Mignard,mythological painting,painter': 'artist',
                    'John Whitcomb': 'artist',
                    'Amedeo Modigliani,landscape art,portrait': 'artist',
                    'Jordan Grimmer': 'artist',
                    'A.J.Casson': 'artist',
                    'Akira Toriyama,mangaka,fantasy': 'artist',
                    'Salvador Dali': 'artist',
                    'Greg Rutkowski': 'artist',
                    'Jeremy Mann': 'artist',
                    'Van Gogh': 'artist',
                    'Monet': 'artist',
                    'Pablo Picasso': 'artist'}.keys()

    concepts_to_remove, concept_labels = preprocess_concepts(concepts_to_remove)
    print(concept_labels)


    all_concept_embeds = save_image_embeddings(concepts_to_remove, root, clip_tokenizer, text_model, args)
    
    concept_checker = ConceptCheckerSingle(concepts_to_remove, concept_labels, clip_tokenizer, text_model, 
                  cos_threshold=args.cos_thr, gpu = args.gpu, text_embeddings=all_concept_embeds)

    accuracy = []
    false_removal = []
    neg_removal = []
    avg_sim_normal = 0

    things = ['cat', 'dog', 'bird', 'car', 'bear', 'bat', 'ball', 'apple', 'banana', 'house', 'tree', 'flower', 'room', 'building', 'wall', 'chicken', 'duck']
    humans = ['man', 'woman', 'child', 'boy', 'girl', 'kid', 'teacher', 'nurse', 'doctor', 'professor', 'guy', 'couple']
    features = []
    for t in things:
        p = 'a photo of a' + t
        feats = concept_checker.embed(p)
        feats = feats/feats.norm(dim=-1, keepdim=True)
        features.append(feats)
    features = torch.stack(features).squeeze(1).mean(0)
    features = features / features.norm(dim=-1, keepdim=True)

    # for t in things:
    #     p = 'a photo of a ' + t
    #     sim = concept_checker.check_concepts(p)
    #     avg_sim_normal += sim
    # avg_sim_normal = avg_sim_normal/len(things)
    # avg_sim_normal = avg_sim_normal.detach().cpu().numpy()

    # avg_sim_prompts = 0
    # for index, row in tqdm.tqdm(prompts.iterrows()):
    #     prompt = row['prompt'].split("\n")[0]
    #     sim = concept_checker.check_concepts(prompt)
    #     avg_sim_prompts += sim
    # avg_sim_prompts = avg_sim_prompts/len(prompts)
    # avg_sim_prompts = avg_sim_prompts.detach().cpu().numpy()

    # print("Average similarity of normal prompts: ", avg_sim_normal)
    # print("Average similarity of prompts: ", avg_sim_prompts)

    output_path = os.path.join('benchmarking results/art-styles')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    accuracy, false_removal, neg_removal = [], [], []
    for index, row in tqdm.tqdm(prompts.iterrows()):
        prompt = row['prompt'].split("\n")[0]
        label = row['concept']
        labels = label.split(',')[0]
        labels = 'none' if labels not in concepts_to_remove else 'artist'
        
        text_embed = concept_checker.embed(prompt).squeeze(0)
        sim = concept_checker.check_concepts(prompt)
        sim = sim.detach().cpu().numpy()
        sim_no_concept = features @ text_embed
        # select max from sim
        max_sim = np.max(sim)
        pred_concept = np.argmax(sim)
        # remove_concept = False
        if max_sim > sim_no_concept and max_sim > 0.55:
            remove_concept = True
        else:
            remove_concept = False

        if remove_concept and labels == 'none':
            false_removal.append((prompt, labels))
        elif remove_concept and labels != 'none':
            accuracy.append(prompt)
        elif not remove_concept and labels == 'none':
            accuracy.append(prompt)
        elif not remove_concept and labels != 'none':
            neg_removal.append((prompt, labels))
        if remove_concept:
            print("Prompt: ", prompt)
            pred_concept = concepts_to_remove[pred_concept]
            path_expert_indx = os.path.join((root % pred_concept).split('images')[0], 'skilled_neuron_wanda/0.05')
            print("Path expert index: ", path_expert_indx)
            neuron_remover = WandaRemoveNeuronsFast(seed=args.seed, path_expert_indx = path_expert_indx, 
                    T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.keep_nsfw,
                    remove_timesteps = args.timesteps, weights_shape = None) 
            neuron_remover.reset_time_layer()
            out, _ = neuron_remover.observe_activation(model, prompt)
            # save images
            prompt_short = prompt[:10] + "_" + labels

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            out_pre = model(prompt, safety_checker=safety_checker_).images[0]

            out_pre = out_pre.resize((256, 256))
            out = out.resize((256, 256))
            new_im = Image.new('RGB', (530, 290))
            new_im.paste(out_pre, (0,40))
            new_im.paste(out, (275,40))
            new_im.save(os.path.join(output_path, f'img_{index}_{prompt_short}.jpg'))

    print("Accuracy: ", len(accuracy))
    print("False removal: ", len(false_removal))
    print("Negative removal: ", len(neg_removal))

    # save the results
    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump({'accuracy': accuracy, 'false_removal': false_removal, 'neg_removal': neg_removal}, f)

        

if __name__ == '__main__':
    main()