import json
import os
import sys
import torch
import tqdm
import types
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from collections import Counter
from expert_predictivity import load_expert_clusters
sys.path.append('moefication')
from helper import modify_ffn_to_experts, initialise_expert_counter
from freq_expert_select import FrequencyMeasure
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from diffusers.models.activations import GEGLU

def main():
    args = utils.Config('experiments/config.yaml', 'modularity')
    args.modularity['skill_type'] = 'moefy'
    args.configure('modularity')
    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication['topk_experts'] = topk
        args.configure('modularity')
    print(f"Topk experts: {args.moefication['topk_experts']}")

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)
    # Change FFNS to a mixture of experts
    model = modify_ffn_to_experts(model, args)

    # load expert clusters
    param_split = os.listdir(os.path.join(args.res_path, 'param_split'))
    expert_clusters = load_expert_clusters(param_split, args.res_path)
    param_names = sorted(expert_clusters.keys())

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        things = f.readlines()
        things = [t.strip() for t in things]
    base_prompts = [f'a {thing}' for thing in things]
    # add an adjective of choice to every element in things list
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in things]

    # Neuron receiver with forward hooks to measure predictivity
    freq_counter = FrequencyMeasure()
    expert_counter, ffn_names_list = initialise_expert_counter(model)
    expert_counter_adj, _ = initialise_expert_counter(model)

    iter = 0
    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)

        freq_counter.clear_counter()
        out, _ = freq_counter.observe_activation(model, ann)
        label_counter = freq_counter.label_counter

        freq_counter.clear_counter()
        out_adj, _ = freq_counter.observe_activation(model, ann_adj)
        label_counter_adj = freq_counter.label_counter

        for i in range(0, len(label_counter), num_geglu):
            gate_timestep, gate_timestep_adj = label_counter[i:i+num_geglu], label_counter_adj[i:i+num_geglu]
            for j, labels, labels_adj in zip(range(num_geglu), gate_timestep, gate_timestep_adj):
                if j > num_geglu:
                    continue
                for label, label_adj in zip(label, label_adj):
                    expert_counter[i//num_geglu][ffn_names_list[j]][label] += 1
                    expert_counter_adj[i//num_geglu][ffn_names_list[j]][label_adj] += 1

        iter += 1
    
    # divide by number of images
    for t in range(args.timesteps):
        for ffn_name in ffn_names_list:
            expert_counter[t][ffn_name] = expert_counter[t][ffn_name].tolist()
            expert_counter_adj[t][ffn_name] = expert_counter_adj[t][ffn_name].tolist()
            expert_indices = expert_clusters[ffn_name]
            for expt_idx in range(max(expert_indices) + 1):
                # get the skilled neurons in the expert
                neurons_in_expert = torch.tensor(expert_clusters[ffn_name]) == expt_idx
                
    

if __name__ == "__main__":
    main()
