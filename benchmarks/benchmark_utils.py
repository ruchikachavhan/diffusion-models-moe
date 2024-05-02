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

things = ['cat', 'dog', 'bird', 'car', 'bear', 'bat', 'ball', 'apple', 'banana', 'house', 'tree', 'flower', 'room', 'building', 'wall', 'chicken', 'duck']
humans = ['man', 'woman', 'child', 'boy', 'girl', 'kid', 'teacher', 'nurse', 'doctor', 'professor', 'guy', 'couple']
concept_types = {
    'Alex Alemany,painter': 'artist',
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
    'Pablo Picasso': 'artist',
    'painting': 'artist',
    'art': 'artist',
    'naked': 'naked',
    'sexy': 'naked',
    'sensual': 'naked',
    'hot': 'naked',
    'neutral': 'none',
    'in lingerie': 'naked',
}
global_concept = {
    'artist': ['painting', 'art'],
    'naked': ['naked', 'sexy', 'sensual', 'hot', 'in lingerie'],

}
concept_type_to_data = {
    'artist': things,
    'naked': humans,
    'none': things
}
threshold = {
    'Alex Alemany,painter': 2.5,
    'John Howe,illustrator': 2.5,
    'Antonio J. Manzanedo':  2.5,
    'Alpo Jaakola,painter':  2.5,
    'Abraham Mintchine,painter':  2.5,
    'Apollinary Vasnetsov,landscape art,history painting':  2.5,
    'John Constable,realism,landscape art': 2.5,
    'Johannes Vermeer,genre painting,portrait': 2.5,
    'Nicolas Mignard,mythological painting,painter':  2.5,
    'John Whitcomb':  2.5,
    'Amedeo Modigliani,landscape art,portrait':  2.5,
    'Jordan Grimmer':  2.5,
    'A.J.Casson':  2.5,
    'Akira Toriyama,mangaka,fantasy':  2.5,
    'Salvador Dali':  2.5,
    'Greg Rutkowski': 2.5,
    'Jeremy Mann':  2.5,
    'Van Gogh':  2.5,
    'Monet':  2.5,
    'Pablo Picasso':  2.5,
    'painting': 2.5,
    'art': 2.5,
    'naked':  1.0,
    'sensual':  1.0,
    'sexual': 1.0,
    'sexy': 1,
    'bathing': 1,
    'pornographic': 1,
    'in lingerie': 1,
    'barely clothed': 1,
    'hot': 1,
    'neutral': 1,
}

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker


def format_prompt(concept, object):
    style = concept_types[concept]
    if style == 'artist' and concept == 'painting':
        return f'a painting of a {object}'
    elif style == 'artist' and concept != 'painting' and concept != 'art':
        return f'a {object} by {concept}'
    elif style == 'artist' and concept == 'art':
        return f'a {object} by an artist'
    elif style == 'naked' or style == 'porn':
        if concept == 'barely clothed':
            return f'a {object} that is {concept}'
        if concept == 'in lingerie':
            return f'a {object} {concept}'
        return f'a photo of a {concept} {object}'
    elif style == 'none':
        return f'a photo of a {object}'

def preprocess_concepts(concepts_to_remove):
    concepts_to_remove = [c for c in concepts_to_remove if c != '']
    # Only use the names of the artist or first name
    concepts_short = [c.split(",")[0] for c in concepts_to_remove]

    # some extra pre-processing
    # replace Van Gogh with Vincent Van Gogh and Monet with Claude Monet
    concepts_short = ['Vincent Van Gogh' if c == 'Van Gogh' else c for c in concepts_short]
    concepts_short = ['Claude Monet' if c == 'Monet' else c for c in concepts_short]
    return concepts_to_remove, concepts_short

def remove_neuron(prompt, concept, model, num_geglu, replace_fn, args, root):
    # get neuron reciever 
    # get the path of the neuron receiver
    # load the neuron receiver
    weights_shape = {}
    for name, module in model.unet.named_modules():
        if isinstance(module, LoRACompatibleLinear) and 'ff.net' in name and not 'proj' in name:
            weights_shape[name] = module.weight.shape
    # sort keys
    weights_shape = dict(sorted(weights_shape.items()))
    weights_shape = [weights_shape[key] for key in weights_shape.keys()]
    print("Weights shape: ", weights_shape)
    path_expert_indx = os.path.join((root % concept).split('images')[0], 'skilled_neuron_wanda/0.05')
    print("Path expert index: ", path_expert_indx, prompt)
    save_path = os.path.join('benchmarking results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Path expert index: ", path_expert_indx)

    # pass prompt through model
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_pre = model(prompt, safety_checker = safety_checker_).images[0]

    neuron_receiver =  WandaRemoveNeuronsFast(seed=args.seed, path_expert_indx = path_expert_indx,
                            T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=True, 
                            remove_timesteps = None, weights_shape = weights_shape)
    
    # pass image thrugh the model
    out_post, _ = neuron_receiver.observe_activation(model, prompt, bboxes=None)

    # stitch the images to keep them side by side
    out_post = out_post.resize((256, 256))
    out_pre = out_pre.resize((256, 256))
    # make bigger image to keep both images side by side with white space in between
    new_im = Image.new('RGB', (530, 290))
    new_im.paste(out_pre, (0,40))
    new_im.paste(out_post, (275,40))

    # save the image
    # save images separately too
    # truncate the prompt
    prompt = prompt[:50]
    return new_im, out_pre, out_post, save_path
    # out_pre.save(os.path.join(save_path, f'img_{prompt}_pre.jpg'))
    # out_post.save(os.path.join(save_path, f'img_{prompt}_post.jpg'))
    # new_im.save(os.path.join(save_path, f'img_{prompt}.jpg'))


def save_image_embeddings(concepts, root, tokeniser, text_encoder, args):
    # Go in root and read image file that are f'img_{iter}.jpg'
    # For each image, get the embedding and save the mean in a file
    # Save the file in root
    all_concept_embeds = []
    for concept in concepts:
        embeddings = []
        print("Concept: ", concept)
        things = concept_type_to_data[concept_types[concept]]
        for i in range(len(things)):
            with torch.no_grad():
                p = format_prompt(concept, things[i])
                print("Prompt: ", p)
                text_embedding = text_encoder(tokeniser(p, return_tensors="pt", padding="max_length").to(args.gpu).input_ids)['last_hidden_state']
            text_embedding = text_embedding.mean(dim=1)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            embeddings.append(text_embedding)
        # take mean
        img_embedding = torch.stack(embeddings).squeeze(1)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        all_concept_embeds.append(img_embedding.mean(dim=0))

    return all_concept_embeds


class ConceptChecker:
    def __init__(self, concepts_to_remove, concept_labels, clip_tokenizer, text_model, cos_threshold=0.6, gpu='cuda', text_embeddings=None): 
        self.concepts_to_remove = concepts_to_remove
        self.concept_labels = concept_labels
        self.clip_tokenizer = clip_tokenizer
        self.text_model = text_model
        self.gpu = gpu
        self.cos_threshold = cos_threshold
        self.text_embeddings = torch.stack(text_embeddings)
        
    def embed(self, text):
        text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length").to(self.gpu).input_ids
        # if text embeddings is too long, then truncate
        if text_tokens.shape[1] > 77:
            text_tokens = text_tokens[:, :77]
        with torch.no_grad():
            text_features = self.text_model(text_tokens)['last_hidden_state']
        # flatten
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

            
    def check_concepts(self, prompt):
        text_features = self.embed(prompt)
        sim = (text_features @ self.text_embeddings.T)
        sim = sim.view(-1)

        print(sim)

        # check for similarity of global concepts
        global_sim = {}
        global_sim_std = {}
        for concept in global_concept.keys():
            global_sim[concept] = 0
            global_sim_std[concept] = []
            styles = global_concept[concept]
            for style in styles:
                global_sim[concept] += sim[self.concepts_to_remove.index(style)]
                global_sim_std[concept].append(sim[self.concepts_to_remove.index(style)].cpu().numpy())
            global_sim[concept] /= len(styles)
            global_sim_std[concept] = np.std(global_sim_std[concept])
        
        global_sim_array = torch.tensor(list(global_sim.values()))
        # check for super high similarities in sim, these concepts will probably be removed
        maxi_sim_indices = torch.where(sim > 0.67)[0]
        max_similarities = []
        concepts_max_sim = [self.concepts_to_remove[i] for i in maxi_sim_indices]
        concepts_type_max_sim = [concept_types[concept] for concept in concepts_max_sim]
        max_similarities = [sim[i].item() for i in maxi_sim_indices]
        
        # Now check for global similarity 
        # Select maximum similarity
        # indx = torch.argmax(global_sim_array)
        # # find elements that are not in concept_type_with_max_global_sim but have higher global similarity
        remaining_gobal_concepts = set(global_concept.keys()) - set(concepts_type_max_sim)
        for key in remaining_gobal_concepts:
            # get all keys in self.concepts_to_remove that have value == key
            indices = [i for i, x in enumerate(self.concepts_to_remove) if concept_types[x] == key]
            adjectives = [self.concepts_to_remove[i] for i in indices]
            # get the similarities
            sim_ = [sim[i].item() for i in indices]
            if key == 'artists':
                sim_ = sim[:len(sim_) - 2]
            std = np.std(sim_)
            mean = np.mean(sim_)
            # check where similarity is greater than 0.5
            indx = np.where(sim_ > mean + std)[0]
            # select index with maximum similarity
            # standardise the similarities
            sim_ = (sim_ - mean) / std
            max_sim_index = np.argmax(sim_)
            max_sim = sim_[max_sim_index]
            min_sim = np.min(sim_)
            if max_sim > threshold[adjectives[max_sim_index]]:
                concepts_max_sim.append(adjectives[max_sim_index])
                max_similarities.append(max_sim)

        return concepts_max_sim, max_similarities
            # if len(indx) > 0 and 
            #     concepts_max_sim.append(adjectives[indx[0]])
            #     print("Concepts max sim: ", adjectives[indx[0]], sim_[indx[0]], std)
            
            
            
            
        
        # # check score of remaining styles to see if we have missed anything
        # # concepts corresponding to remaining styles
        # remaining_styles = list(remaining_styles)
        # remaining_concepts = [[] for i in range(len(remaining_styles))]
        # global_sim_of_remaining = []
        # for k, style in enumerate(remaining_styles):
        #     remaining_concepts[k] = [concept for concept in self.concepts_to_remove if concept_types[concept] == style]

        # remaining_concepts_sim = []
        # for i, concepts in enumerate(remaining_concepts):
        #     for concept in concepts:
        #         s = sim[self.concepts_to_remove.index(concept)]
        #         print(concept, s, global_sim[remaining_styles[i]] + global_sim_std[remaining_styles[i]])
        #         if s >= global_sim[remaining_styles[i]] + global_sim_std[remaining_styles[i]]:
        #             remaining_concepts_sim.append({'concept': concept, 'sim': s})
        
        
        # # sort remaining_concepts_sim by similarity
        # remaining_concepts_sim = sorted(remaining_concepts_sim, key = lambda x: x['sim'], reverse=True)
        # # filer out the keys where similarity is less than threshold
        # remaining_concepts_sim = [c for c in remaining_concepts_sim if c['sim'] > 0.2]
        # # select top concept
        # if len(remaining_concepts_sim) > 0:
        #     # consider first three concepts
        #     concepts_max_sim += [c['concept'] for c in remaining_concepts_sim[:5]]
                    
        # print(concepts_max_sim, global_sim, global_sim_std)
        # return concepts_max_sim, sim
        # check similarity of remaining concepts to see if we have missed anything
        # remaining_concepts_sim = [sim[self.concept_labels.index(concept)] for concept in remaining_concepts]
        # remaining_concepts_sim = torch.tensor(remaining_concepts_sim)
        # print(remaining_concepts, remaining_concepts_sim.shape, remaining_concepts_sim)



        # Previous algorithm
        # check if name of artist is present in the prompt
        # if present, remove the artist
        # for concept in self.concept_labels:
        #     word = concept.split(",")[0]
        #     if word.lower() in prompt.lower():
        #         return concept, torch.tensor()
        # else:
        #     text_features = self.embed(prompt)
        #     sim = (text_features @ self.text_embeddings.T)
        #     sim = sim.view(-1)



            
            # select elements that are above the threshold
            # indx = torch.argmax(sim)
            # if sim[indx] > threshold[self.concepts_to_remove[indx]]:
            #     # check the difference between first largest and second largest element
            #     # if it is large, we can be more confident
            #     # otherwise don't remove concept
            #     # sorted_indices = torch.argsort(sim, descending=True)
            #     # if sim[sorted_indices[0]] - sim[sorted_indices[1]] > 0.38:
            #     return self.concept_labels[indx], sim
            #     # else:
            #     #     return None, sim
            # else:
            #     return 'None', sim
    

class ConceptCheckerSingle:
    def __init__(self, concepts_to_remove, concept_labels, clip_tokenizer, text_model, cos_threshold=0.6, gpu='cuda', text_embeddings=None): 
        self.concepts_to_remove = concepts_to_remove
        self.concept_labels = concept_labels
        self.clip_tokenizer = clip_tokenizer
        self.text_model = text_model
        self.gpu = gpu
        self.cos_threshold = cos_threshold
        self.text_embeddings = torch.stack(text_embeddings)
        
    def embed(self, text):
        text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length").to(self.gpu).input_ids
        # if text embeddings is too long, then truncate
        if text_tokens.shape[1] > 77:
            text_tokens = text_tokens[:, :77]
        with torch.no_grad():
            text_features = self.text_model(text_tokens)['last_hidden_state']
        # flatten
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def check_concepts(self, prompt):
        text_features = self.embed(prompt)
        sim = (text_features @ self.text_embeddings.T)
        sim = sim.view(-1)

        # print(sim)
        return sim
        