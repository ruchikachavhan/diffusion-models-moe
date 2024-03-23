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
from mod_utils import get_prompts, update_set_diff
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

def main():
    args = utils.Config('experiments/mod_config_moefy_compare.yaml', 'modularity')

    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication = {}
        args.moefication['topk_experts'] = topk
        args.configure('modularity')
    print(f"Topk experts: {args.moefication['topk_experts']}")


    # Model
    model, _ = utils.get_sd_model(args)
    model = model.to(args.gpu)
    # Change FFNS to a mixture of experts
    model, ffn_names_list, num_experts_per_ffn = modify_ffn_to_experts(model, args)

    # Dataset from things.txt
    base_prompts, adj_prompts, sym_diff = get_prompts(args)
  
    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)

    # Neuron receiver with forward hooks to measure predictivity
    freq_counter = GetExperts(args.seed, args.timesteps, args.n_layers, num_experts_per_ffn, ffn_names_list, keep_nsfw=args.modularity['keep_nsfw'])
    iter = 0

    set_diff = {}
    for t in range(args.timesteps):
        set_diff[t] = {}
        for l, ffn_name in enumerate(ffn_names_list):
            set_diff[t][l] = []

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
        
        for t in range(args.timesteps):
            for l, ffn_name in enumerate(ffn_names_list):
                diff = update_set_diff(set(label_counter_adj[t][l]), set(label_counter[t][l]), symm=sym_diff)
                set_diff[t][l] += diff 
                print(f"Set difference at time step {t} and layer {l}: {len(set_diff[t][l])}")

        iter += 1

    # Compare most commonly selected experts
   
    for t in range(args.timesteps):
        for l in range(args.n_layers):
            print("saving in ", os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'))
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                counter = Counter(set_diff[t][l])
                # select top - 70% of the experts]
                topk = int(1.0 * len(counter))
                # do argmax in descending order for topk
                # argsort in descending order
                counter = counter.most_common(topk)

                # select the topk experts that are activated by more than 70% of the samples
                activated_experts = []
                for i in range(len(counter)):
                    if counter[i][1] >= int(args.modularity['condition']['skill_ratio'] * len(adj_prompts)):
                        activated_experts.append(counter[i][0])

                activated_experts = [int(expert) for expert in activated_experts]
                print(f"Activated experts at time step {t} and layer {l}: {activated_experts}")
                
                json.dump(activated_experts, f)
        

if __name__ == "__main__":
    main()

