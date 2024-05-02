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

def get_similarities(root, model, processor, args, adjective, prompts):

    # read images in root
    removed_prefix = 'img_{}_adj.jpg'
    base_prefix = 'img_{}.jpg'

    # read images
    avg_similarity = 0
    for i in range(len(prompts)):
        print("Prompt: ", prompts[i])
        # Read base and removed images
        base_img = Image.open(os.path.join(root, base_prefix.format(i)))
        removed_img = Image.open(os.path.join(root, removed_prefix.format(i)))
        # Transform images to tensor
        base_img = processor(images=base_img, return_tensors="pt").to(args.gpu)
        removed_img = processor(images=removed_img, return_tensors="pt").to(args.gpu)
        # get the embeddings
        with torch.no_grad():
            base_embedding = model.get_image_features(**base_img)
            removed_embedding = model.get_image_features(**removed_img)
        # get the similarity score
        base_embedding = base_embedding / base_embedding.norm(dim=-1, keepdim=True)
        removed_embedding = removed_embedding / removed_embedding.norm(dim=-1, keepdim=True)

        similarity = (base_embedding @ removed_embedding.T)
        print("Similar images: ", similarity)
        avg_similarity += similarity.item()

    print("Similar images: ", avg_similarity/len(prompts))



def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')
    adjective = args.modularity['adjective']
    test_name = args.modularity['condition']['name']
    folder = '0.3/dof_9_conf_0.001' if args.modularity['condition']['name'] == 't_test' else '0.01'

    base_prompts, adj_prompts, _ = get_prompts(args)
    print(adj_prompts)

    # read validation set
    with open(f'modularity/datasets/val_things_{adjective}.txt') as f:
        val_objects = f.readlines()

    # Load the CLIP model
    model_id = "openai/clip-vit-large-patch14"
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


    root = f'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/{adjective}/skilled_neuron_{test_name}/{folder}/remove_neurons'
    get_similarities(root, model, processor, args, adjective, adj_prompts)

        

if __name__ == '__main__':
    main()