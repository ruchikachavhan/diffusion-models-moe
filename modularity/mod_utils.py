import os
import sys
import torch
import numpy as np
from nltk.util import everygrams
from transformers import CLIPTextModel, CLIPTokenizer

def get_prompts(args):
    # Dataset from things.txt
    # read things.txt
    f_name = args.modularity['file_name']
    with open(f'modularity/datasets/{f_name}.txt', 'r') as f:
        things = f.readlines()
        things = [t.strip() for t in things]
    # add an adjective of choice to every element in things list
    adjectives = [args.modularity['adjective']]
    if len(adjectives) == 1:
        base_prompts = [f'a {thing}' for thing in things]

        if adjectives[0] in ['white', 'black', 'jumping', 'running']:
            # simply add the adjective before the thing
            base_prompts = [f'a {thing}' for thing in things]
            adj_prompts = [f'a {adjectives[0]} {thing}' for thing in things]
            remove_token_idx = [2]

        elif adjectives[0] in ['under a tree', 'on a table', 'behind a tree', 'on the left of a tree', 'near a tree', 'in a room', 'on the street']:
            # add the adjective after the thing
            adj_prompts = [f'a {thing} {adjectives[0]}' for thing in things]
            remove_token_idx = [3, 4, 5]

        elif adjectives[0] in ['painting']:
            # change base prompt to 'a photo of a {thing}'
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a {adjectives[0]} of a {thing}' for thing in things]
            remove_token_idx = [2, 3, 4]

        elif adjectives[0] in ['bloody red']:
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a photo of a {adjectives[0]} {thing} carcass' for thing in things]
            args.modularity['condition']['is_nsfw'] = True
            args.modularity['keep_nsfw'] = True
            remove_token_idx = [5, 6, 8, 9]
            
        elif adjectives[0] in ['naked']:
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a photo of a {adjectives[0]} {thing}' for thing in things]
            args.modularity['condition']['is_nsfw'] = True
            args.modularity['keep_nsfw'] = True
            remove_token_idx = [5]
        
        elif adjectives[0] in ['Van Gogh', 'Monet', 'Greg Rutkowski', 'Jeremy Mann', 'Salvador Dali', 'Pablo Picasso', 'manga']:
            base_prompts = [f'a {thing}' for thing in things]
            adj_prompts = [f'a {thing} in the style of {adjectives[0]}' for thing in things]
            remove_token_idx = [3, 4, 5, 6, 7, 8]

        elif adjectives[0] in ['gender']:
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a photo of a {thing}' for thing in things]

        elif adjectives[0] in ['scene_removal_cat']:
            base_prompts = [f'a {thing}' for thing in things]
            adj_prompts = [f'a {thing} with a cat' for thing in things]
            remove_token_idx = [3, 4, 5]
        elif adjectives[0] in ['cassette player']:
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a photo of a {adjectives[0]}' for _ in things]
            remove_token_idx = [3, 4, 5]
        
        elif adjectives[0] in ['memorize']:
            base_prompts = ['' for _ in things]
            adj_prompts = [thing for thing in things]
            remove_token_idx = [3, 4, 5]

    elif len(adjectives) == 2:
        # consider the first adjective as base prompt
        if adjectives[0] in ['white', 'black']:
            base_prompts = [f'a {adjectives[0]} {thing}' for thing in things]
            adj_prompts = [f'a {adjectives[1]} {thing}' for thing in things]
        elif adjectives[0] in ['under a tree', 'on the street'] and adjectives[1] != 'painting' :
            base_prompts = [f'a {thing} {adjectives[1]}' for thing in things]
            adj_prompts = [f'a {thing} {adjectives[0]} {adjectives[1]}' for thing in things]
        elif adjectives[0] in ['under a tree', 'on the street'] and adjectives[1] == 'painting':
            base_prompts = [f'a photo of a {thing}' for thing in things]
            adj_prompts = [f'a painting of a {thing} {adjectives[0]}' for thing in things]
    else:
        raise ValueError("Only 1 or 2 adjectives are allowed")

    if not args.modularity['single_sample_test']:
        return base_prompts, adj_prompts, True if len(adjectives) == 2 else False, remove_token_idx
    else:
        return [base_prompts[7]], [adj_prompts[7]], True if len(adjectives) == 2 else False, remove_token_idx

def update_set_diff(set1, set2, symm=False):
    if symm:
        return set1.symmetric_difference(set2)
    else:
        return set1.difference(set2)
    
def similarity_ngrams_concept(prompt, concept, tokenizer, text_encoder, args):

    # pass ngrams and concept into CLIP text encoder
    ngrams_tkns = tokenizer(prompt).input_ids
    
    concept_tkns = tokenizer(concept).input_ids

    ngrams_feat = [text_encoder(torch.tensor([49406, ngrams_tkns[i], 49407]).unsqueeze(0)).pooler_output for i in range(len(ngrams_tkns)) if ngrams_tkns[i] not in [49406, 49407]]
    ngrams_feat = torch.cat(ngrams_feat, dim=0)
    concept_feat = [text_encoder(torch.tensor([49406, concept_tkns[i], 49407]).unsqueeze(0)).pooler_output for i in range(len(concept_tkns)) if concept_tkns[i] not in [49406, 49407]]
    concept_feat = torch.cat(concept_feat, dim=0)

    # normalize the features
    ngrams_feat = ngrams_feat / ngrams_feat.norm(dim=-1, keepdim=True).squeeze(0)
    concept_feat = concept_feat / concept_feat.norm(dim=-1, keepdim=True).squeeze(0)

    # Calculate cosine similarity between the ngrams and concept
    similarity = (ngrams_feat @ concept_feat.T).squeeze(1).detach().cpu().numpy()
    # Select the top 5 ngrams with highest similarity
    indices = np.argsort(similarity)[::-1][:int(0.3*len(similarity))].tolist()
    # indices = [i for i in indices if i != 0]
    return indices



