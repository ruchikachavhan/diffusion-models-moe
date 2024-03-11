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

def update_expert_counter(args, ffn_names_list, num_experts_per_ffn, init=False, label_counter=None, expert_counter=None, num_samples=None):
    if init:
        expert_counter = {}
        for t in range(args.timesteps):
            expert_counter[t] = {}
            for ffn_name in ffn_names_list:
                expert_counter[t][ffn_name] = [0] * num_experts_per_ffn[ffn_name]
    else:
        # add to expert counter
        for t in range(args.timesteps):
            for ffn_name in ffn_names_list:
                for expert in range(num_experts_per_ffn[ffn_name]):
                    expert_counter[t][ffn_name][expert] += label_counter[t][ffn_names_list.index(ffn_name)][expert] / num_samples
    return expert_counter

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

    # bad solution for averaging but had to do it
    expert_counter = update_expert_counter(args, ffn_names_list, num_experts_per_ffn, init=True)
    expert_counter_adj = update_expert_counter(args, ffn_names_list, num_experts_per_ffn, init=True)

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

        out_adj, _ = freq_counter.observe_activation(model, ann_adj, bboxes=bb_coordinates_layer_adj[ann_adj] if args.modularity['bounding_box'] else None)
        # save image
        out_adj.save(f'test_images/test_image_adj.png')
        label_counter_adj = freq_counter.label_counter.copy()

        # add to expert counter
        expert_counter = update_expert_counter(args, ffn_names_list, num_experts_per_ffn, 
                                               label_counter=label_counter, expert_counter=expert_counter, num_samples=len(base_prompts))
        expert_counter_adj = update_expert_counter(args, ffn_names_list, num_experts_per_ffn, 
                                                label_counter=label_counter_adj, expert_counter=expert_counter_adj, num_samples=len(adj_prompts))

        iter += 1

    # Compare most commonly selected experts
    set_diff = {}
    for t in range(args.timesteps):
        set_diff[t] = {}
        for l, ffn_name in enumerate(ffn_names_list):
            set_diff[t][l] = []

    for t in range(args.timesteps):
        for l, ffn_name in enumerate(ffn_names_list):
            list_freq1 = expert_counter[t][ffn_name]
            list_freq2 = expert_counter_adj[t][ffn_name]
            # select top - 70% of the experts
            topk = int(0.6 * len(list_freq1))
            # do argmax in descending order for topk
            # argsort in descending order 
            list_freq1 = np.argsort(list_freq1)[::-1][:topk]
            list_freq2 = np.argsort(list_freq2)[::-1][:topk]

            # find the difference in the sets
            set_diff[t][l] = set(list_freq2) - set(list_freq1)
            set_diff[t][l] = list(int(i) for i in set_diff[t][ffn_name])
            print(f"Set difference at time step {t} and layer {ffn_name}: {set_diff[t][l]}")
            # save the difference
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(set_diff[t][l], f)


if __name__ == "__main__":
    main()

