import torch
import numpy as np
import os
import json
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel

def preprocess_concepts(concepts_to_remove):
    concepts_to_remove = [c for c in concepts_to_remove if c != '']
    # Only use the names of the artist or first name
    concepts_short = [c.split(",")[0] for c in concepts_to_remove]

    # some extra pre-processing
    # replace Van Gogh with Vincent Van Gogh and Monet with Claude Monet
    concepts_short = ['Vincent Van Gogh' if c == 'Van Gogh' else c for c in concepts_short]
    concepts_short = ['Claude Monet' if c == 'Monet' else c for c in concepts_short]
    return concepts_to_remove, concepts_short

class BaseConceptChecker:
    def __init__(self, device, objects, concepts_to_remove):
        print("Removing concepts", concepts_to_remove)
        self.concepts_to_remove = concepts_to_remove
        self.device = device
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.list_of_objects = objects
        self.all_concept_features = self.embed_all_objects()
    
    def format_prompt(self, obj, concept=None):
        return NotImplementedError
    
    def no_concept_features(self, things):
        features = []
        for t in things:
            p = 'a photo of a' + t
            feats = self.text_model(self.clip_tokenizer(p, return_tensors="pt", padding="max_length").to(self.device).input_ids)['last_hidden_state']
            feats = feats.mean(dim=1)
            feats = feats/feats.norm(dim=-1, keepdim=True)
            features.append(feats)
        features = torch.stack(features).squeeze(1).mean(0)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def embed_all_objects(self):
        all_concept_features = {}
        for concept in self.concepts_to_remove:
            object_features = []
            for obj in self.list_of_objects:
                p = self.format_prompt(obj, concept)
                with torch.no_grad():
                    tokens = self.clip_tokenizer(p, return_tensors="pt", padding="max_length").to(self.device).input_ids
                    text_embedding = self.text_model(tokens)['last_hidden_state']
                # Take mean because this returns embedding for all tokens
                text_embedding = text_embedding.mean(dim=1)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                object_features.append(text_embedding)
            object_features = torch.stack(object_features).squeeze(1)
            # normalise 
            object_features = object_features / object_features.norm(dim=-1, keepdim=True)
            # take mean over all objects
            object_features = object_features.mean(0)
            all_concept_features[concept] = object_features

        return all_concept_features

    def similarity(self, text):
        with torch.no_grad():
            text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length").to(self.device).input_ids
            if text_tokens.shape[1] > 77:
                text_tokens = text_tokens[:, :77]
            text_features = self.text_model(text_tokens)['last_hidden_state']
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarities = {}
        for c in self.all_concept_features.keys():
            # find similarity with all objects
            sim = (text_features @ self.all_concept_features[c].T)
            sim = sim.view(-1)
            similarities[c] = sim
        return similarities, text_features

    def decide(self, text):
        return NotImplementedError

class NudityChecker(BaseConceptChecker):
    def __init__(self, device, objects, neg_objects, concepts_to_remove=['naked', 'sexy']):
        super().__init__(device, objects, concepts_to_remove)

        self.concepts_to_remove, self.concept_labels = preprocess_concepts(concepts_to_remove)
        print("Nudity Checker", self.concepts_to_remove, self.concept_labels)

        self.anchor_file = 'modularity/datasets/i2p_prompts_seed_0_CompVis_stable-diffusion-v1-4.txt'
        self.anchor_features = self.embed_anchor_prompts()
        self.neg_objects = neg_objects
        self.no_concept_embeds = self.no_concept_features(self.neg_objects)

    def format_prompt(self, obj, c):
        return f"a photo of a {c} {obj}"

    def embed_anchor_prompts(self):
    # take random prompts from the dataset
        # read file 
        with open(self.anchor_file, 'r') as f:
            anchor_prompts = f.readlines()
        anchor_prompts = np.random.choice(anchor_prompts, 100)
        anchor_features = []
        for p in anchor_prompts:
            text_tokens = self.clip_tokenizer(p, return_tensors="pt", padding="max_length").to(self.device).input_ids
            if text_tokens.shape[1] > 77:
                text_tokens = text_tokens[:, :77]
            features = self.text_model(text_tokens)['last_hidden_state']
            features = features.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
            anchor_features.append(features)
        anchor_features = torch.stack(anchor_features).squeeze(1).mean(0)
        return anchor_features

    def decide(self, text):
        sim, text_features = self.similarity(text)
        # get max similarity
        max_sim = max(sim.values())
        sim_no_concept = text_features @ self.no_concept_embeds.T
        sim_prompts = text_features @ self.anchor_features.T
       
        remove_concept = False
        if max_sim > sim_no_concept:
            remove_concept = True
        elif sim_prompts > sim_no_concept:
            remove_concept = True
        else:
            remove_concept = False
        return 'naked' if remove_concept else 'none'


# concepts_to_remove = open('modularity/datasets/famous_artists.txt').read().split('\n')
art_styles = {'Alex Alemany,painter': 'artist',
        # 'John Howe,illustrator': 'artist',
        # 'Antonio J. Manzanedo': 'artist',
        # 'Alpo Jaakola,painter': 'artist',
        # 'Abraham Mintchine,painter': 'artist',
        # 'Apollinary Vasnetsov,landscape art,history painting': 'artist',
        # 'John Constable,realism,landscape art': 'artist',
        # 'Johannes Vermeer,genre painting,portrait': 'artist',
        # 'Nicolas Mignard,mythological painting,painter': 'artist',
        # 'John Whitcomb': 'artist',
        # 'Amedeo Modigliani,landscape art,portrait': 'artist',
        # 'Jordan Grimmer': 'artist',
        # 'A.J.Casson': 'artist',
        # 'Akira Toriyama,mangaka,fantasy': 'artist',
        # 'Salvador Dali': 'artist',
        # 'Greg Rutkowski': 'artist',
        # 'Jeremy Mann': 'artist',
        'Van Gogh': 'artist',
        'Monet': 'artist',
        'Pablo Picasso': 'artist'}.keys()
    
class ArtStyleChecker(BaseConceptChecker):
    def __init__(self, device, objects, neg_objects, concepts_to_remove=art_styles):
        super().__init__(device, objects, concepts_to_remove)
        self.concepts_to_remove, _ = preprocess_concepts(concepts_to_remove)
        print("Art Style Checker", self.concepts_to_remove)
        self.neg_objects = neg_objects
        self.no_concept_embeds = self.no_concept_features(self.neg_objects)
        self.threshold = 0.55

    def format_prompt(self, obj, c):
        return f"a {obj} by {c}"
    
    def decide(self, text):
        sim, text_features = self.similarity(text)
        # get max similarity
        max_sim = max(sim.values())
        max_sim_index = max(sim, key=sim.get)
        sim_no_concept = text_features @ self.no_concept_embeds.T
        pred_concept = max_sim_index

        remove_concept = False
        if max_sim > sim_no_concept and max_sim > self.threshold:
            remove_concept = True
        else:
            remove_concept = False

        return pred_concept if remove_concept else 'none'

        
class MemorizedPromptChecker:
    def __init__(self, device, objects, neg_objects, concepts_to_remove=['memorize']):
        # read parquet file with prompts from modularity dataset
        self.device = device
        # self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        prompts = pd.read_parquet('modularity/datasets/sdv1_bb_edge_groundtruth.parquet')
        self.prompts = prompts['caption'].values.tolist()
        # print(self.prompts)
        # # embed all the prompts
        # if not os.path.exists('modularity/datasets/memorised_prompts.pt'):
        #     all_concept_features = []
        #     print("Embedding all prompts")
        #     for p in prompts:
        #         print(p)
        #         text_tokens = self.clip_tokenizer(p, return_tensors="pt", padding="max_length").to(self.device).input_ids
        #         if text_tokens.shape[1] > 77:
        #             text_tokens = text_tokens[:, :77]
        #         features = self.text_model(text_tokens)['last_hidden_state']
        #         features = features.mean(dim=1)
        #         features = features / features.norm(dim=-1, keepdim=True)
        #         features = features.cpu().detach()
        #         all_concept_features.append(features)
        #         del features
            
        #     self.all_concept_features = torch.stack(all_concept_features).squeeze(1)
        #     # save the embeddings in a pt file
        #     torch.save(self.all_concept_features, 'modularity/datasets/memorised_prompts.pt')
        # else:
        #     self.all_concept_features = torch.load('modularity/datasets/memorised_prompts.pt')

        self.threshold = 0.9
    
    def format_prompt(self, obj, concept=None):
        return obj
    
    def similarity(self, text):
        with torch.no_grad():
            text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding="max_length").to(self.device).input_ids
            if text_tokens.shape[1] > 77:
                text_tokens = text_tokens[:, :77]
            text_features = self.text_model(text_tokens)['last_hidden_state']
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # find similarity with all objects
        sim = (text_features @ self.all_concept_features.T)
        sim = sim.view(-1)
        return sim, text_features
    
    def decide(self, text):
        if text in self.prompts:
            return 'memorize'
        else:
            return 'none'
        # # sim, text_features = self.similarity(text)
        # print(sim)
        # # If sim is high, then remove the concept
        # remove_concept = False
        # if sim > self.threshold:
        #     remove_concept = True
        # else:
        #     remove_concept = False
        # return 'memorize' if remove_concept else 'none'