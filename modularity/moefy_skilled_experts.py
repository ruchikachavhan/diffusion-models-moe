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
from greater import load_expert_clusters
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from neuron_receivers import GetExperts
sys.path.append(os.getcwd())
import utils
from scipy import stats
from collections import Counter


def update_table(table, freq_base, freq_adj):
    # update the contingency table
    # 0, 0 is the when freq base and freq adj is True
    # 0, 1 is when freq base is False and freq adj is True
    # 1, 0 is when freq base is Trye and freq adj is False
    # 1, 1 is when freq base and freq adj is False
    table[0, 0] += (freq_base & freq_adj).sum() 
    table[0, 1] += (freq_base & ~freq_adj).sum()
    table[1, 0] += (~freq_base & freq_adj).sum()
    table[1, 1] += (~freq_base & ~freq_adj).sum()
    return table

def check_condition(val, args):
    return val > args.modularity['condition']['skill_ratio']

def main():
    args = utils.Config('experiments/mod_config_moefy_compare.yaml', 'modularity')
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
    model, ffn_names_list, num_experts_per_ffn = modify_ffn_to_experts(model, args)

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

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    # Neuron receiver with forward hooks to measure predictivity
    freq_counter = GetExperts(args.seed, args.timesteps, args.n_layers, num_experts_per_ffn, ffn_names_list)
    iter = 0

    expert_diff, union_counter = {}, {}
    for t in range(args.timesteps):
        expert_diff[t] = {}
        union_counter[t] = {}
        for l in range(args.n_layers):
            expert_diff[t][l] = []
            union_counter[t][l] = []

    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)

        freq_counter.reset()
        out, _ = freq_counter.observe_activation(model, ann, bboxes=bb_coordinates_layer_base[ann] if args.modularity['bounding_box'] else None)
        # save image
        out.save(f'test_images/test_image_base.png')
        label_counter = freq_counter.label_counter.copy()
            
        freq_counter.reset()
        out_adj, _ = freq_counter.observe_activation(model, ann_adj, bboxes=bb_coordinates_layer_adj[ann_adj+'\n'] if args.modularity['bounding_box'] else None)
        # save image
        out_adj.save(f'test_images/test_image_adj.png')
        label_counter_adj = freq_counter.label_counter.copy()

        iter += 1

        # for each timestep and layer, select top 70% most frequently selected experts
        # this corresponds to selecting experts that have been chosen by more than 70% of the tokens
        for t in range(args.timesteps):
            for l in range(args.n_layers):
                # select the top 70% most frequently selected experts
                expert_base = label_counter[t][l]
                expert_adj = label_counter_adj[t][l]
                # symmetric set difference
                diff = list(set(expert_adj) - set(expert_base))
                expert_diff[t][l] += diff
                # count how many times an expert occured in the union of the two sets
                union_counter[t][l] = Counter(expert_diff[t][l])

                  
    # based on frequency of that expert in the union, select the expert
    for t in range(args.timesteps):
        for l in range(args.n_layers): 
            # sort the counter keys based on values
            union_counter[t][l] = dict(sorted(union_counter[t][l].items(), key=lambda item: item[1], reverse=True))
            # save top 80% most frequently selected experts
            skilled_experts = list(union_counter[t][l].keys())[:int(len(union_counter[t][l]) * 0.5)]
            skilled_experts = [int(expert) for expert in skilled_experts]
            print(f'timestep: {t}, layer: {l}, skilled experts: {skilled_experts}')
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(skilled_experts, f)
        iter += 1


if __name__ == "__main__":
    main()

