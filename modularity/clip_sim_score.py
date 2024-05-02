# Benchmark CLIP similarity against human evaluation of similarity to test for quality of images and removal of concept
import os
import sys
import json
import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, AutoProcessor
from mod_utils import get_prompts
sys.path.append(os.getcwd())
import utils
sys.path.append('sparsity')
from eval_coco import CLIPModelWrapper
from paired_t_test import critical_value_ranges

def get_similarities(root, base_prompts, adj_prompts, model, processor, tokenizer, args, adjective, dof, conf_int, adj_text):
     # get embeddings of base prompts
    base_prompt_embeddings = {}
    for iter in range(len(base_prompts)):
        prompt = base_prompts[iter]
        text = tokenizer(prompt, padding="max_length", return_tensors="pt").to(args.gpu)
        text = model.get_text_features(**text)
        base_prompt_embeddings[iter] = text/text.norm(dim=-1, keepdim=True)
    
    # for every sample create a list of scores for every confidence intervals
    scores_dict, scores_adj = {}, {}
    for s in range(len(base_prompts)):
        scores_dict[s] = {}
        scores_adj[s] = {}

    for conf in conf_int:
        # read results from remove_neurons_folder
        root_ = root % (f'dof_{dof}_conf_{conf}')
        files = os.listdir(root_)
        files.sort()
        after_removal, before_removal = [], []
        
        for iter in range(len(base_prompts)):
            im = Image.open(os.path.join(root_, 'img_{}_adj.jpg'.format(iter)))
            # pass through CLIP image encoder
            img = processor(images=im, return_tensors="pt").to(args.gpu)
            with torch.no_grad():
                img_embedding = model.get_image_features(**img)
            # get the similarity score
            image_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            similarity = (base_prompt_embeddings[iter] @ image_embedding.T)
            print("Similarity: ", similarity)
            scores_dict[iter][conf] = similarity.item()

            # get the similarity score for adjective
            similarity_adj = (adj_text @ image_embedding.T)
            print("Similarity adj: ", similarity_adj)
            scores_adj[iter][conf] = similarity_adj.item()
    # print as dataframe
    import pandas as pd
    df = pd.DataFrame(scores_dict)
    print(df)
    df = pd.DataFrame(scores_adj)
    print(df)


def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')
    adjective = args.modularity['adjective']


    base_prompts, adj_prompts, _ = get_prompts(args)
    print(base_prompts)

    # read validation set
    with open(f'modularity/datasets/val_things_{adjective}.txt') as f:
        val_objects = f.readlines()

    # Load the CLIP model
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(args.gpu)
    model = CLIPModel.from_pretrained(model_id).to(args.gpu)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    base_root = args.modularity['img_save_path']
    base_prefix = 'base_{}'
    dof, conf_int, dof_critical_values = critical_value_ranges()
    dof = len(base_prompts) - 1
    print(dof_critical_values) 

    # encode the adjective
    adj_prompt = f'a painting in the style of {adjective}'
    adj_text = tokenizer(adj_prompt, padding="max_length", return_tensors="pt").to(args.gpu)
    adj_text = model.get_text_features(**adj_text)
    adj_text = adj_text / adj_text.norm(dim=-1, keepdim=True)


    root = f'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/{adjective}/skilled_neuron_t_test/0.3/%s/remove_neurons'
    get_similarities(root, base_prompts, adj_prompts, model, processor, tokenizer, args, adjective, dof, conf_int, adj_text)

    root = f'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/{adjective}/skilled_neuron_t_test/0.3/%s/remove_neurons_val'
    get_similarities(root, val_objects, val_objects, model, processor, tokenizer, args, adjective, dof,conf_int, adj_text)

        

if __name__ == '__main__':
    main()
