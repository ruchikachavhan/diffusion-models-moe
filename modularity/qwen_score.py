from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import cv2
import numpy as np
import json
from PIL import Image
from mod_utils import get_prompts
torch.manual_seed(1234)
import sys
import os
sys.path.append(os.getcwd())
import utils

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')

    qwen_oracle = True

    adjective = args.modularity['adjective']

    base_prompts, adj_prompts, _ = get_prompts(args)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    if qwen_oracle:
        save_path = args.modularity['img_save_path']
        gt_labels = torch.tensor([0, 1]).to(args.gpu)
        prefix = ['base_{}', 'adj_{}']
    else:
        save_path=args.modularity['remove_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path']
        gt_labels = torch.tensor([1, 0]).to(args.gpu)
        prefix = ['img_{}', 'img_{}_adj']


    # load path to images 
    images = os.listdir(save_path)

    # separate base and adj images
    base_images = sorted([img for img in images if 'adj' not in img])
    adj_images = sorted([img for img in images if 'adj' in img])
    print(base_images)
    wrong_samples =[]

    avg_score = 0

    # get the embeddings for the base images
    for iter in range(len(adj_prompts)):
        img_base = os.path.join(save_path, prefix[0].format(iter) + '.jpg')
        img_adj = os.path.join(save_path, prefix[1].format(iter) + '.jpg')

        base_text_feat = 'a photo of ' + base_prompts[iter]
        object = base_prompts[iter].split(' ')[-1]

        print("Base:", base_text_feat)
        query = tokenizer.from_list_format([
        {"image": img_adj},
        {"text": "What is this?"},
         ])
        
        with torch.no_grad():
            response, history = model.chat(tokenizer, query=query, history=None)
            print("Response:", response)
            # response, history = model.chat(tokenizer, 'Is the' + object + adjective + "? Answer in yes or no", history=history)
            response, history = model.chat(tokenizer, 'Is this a painting of a ' + object + "? Answer in yes or no", history=history)
            print("Response:", response)
            
            
if __name__ == '__main__':
    main()
        

