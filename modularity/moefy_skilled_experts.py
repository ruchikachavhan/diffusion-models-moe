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
from neuron_receivers import FrequencyMeasure
sys.path.append(os.getcwd())
import utils
from scipy import stats


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

    # Neuron receiver with forward hooks to measure predictivity
    freq_counter = FrequencyMeasure(args.timesteps, args.n_layers, num_experts_per_ffn, ffn_names_list)
    iter = 0

    contingency_table = {}
    for t in range(args.timesteps):
        contingency_table[t] = {}
        for ffn_name in ffn_names_list:
            contingency_table[t][ffn_name] = {}
            for expt_idx in range(max(expert_clusters[ffn_name]) + 1):
                contingency_table[t][ffn_name][expt_idx] = np.zeros((2, 2))

    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)

        freq_counter.reset()
        out, _ = freq_counter.observe_activation(model, ann)
        label_counter = freq_counter.label_counter

        freq_counter.reset()
        out_adj, _ = freq_counter.observe_activation(model, ann_adj)
        label_counter_adj = freq_counter.label_counter

        # update the contigency tables for each timestep
        for t in range(args.timesteps):
            for ffn_name in ffn_names_list:
                expert_indices = expert_clusters[ffn_name]
                for expt_idx in range(max(expert_indices) + 1):
                    # frequency of expert in base prompt and adj prompt
                    # For every sampe, frequency counter returns the average number of times expert was selected by all tokens. 
                    # If more than 50% of the tokens in the latent space select the expert, then the expert is considered skilled
                    freq_base = check_condition(label_counter[t][ffn_names_list.index(ffn_name)][expt_idx], args)
                    freq_adj = check_condition(label_counter_adj[t][ffn_names_list.index(ffn_name)][expt_idx], args)
                    # update the contingency table
                    contingency_table[t][ffn_name][expt_idx] = update_table(contingency_table[t][ffn_name][expt_idx], freq_base, freq_adj)
                      
        iter += 1

    # print the contingency tables
    for t in range(args.timesteps):
        for ffn_name in ffn_names_list:
            for expt_idx in range(max(expert_clusters[ffn_name]) + 1):
                print(f"Contingency table for timestep {t}, ffn {ffn_name}, expert {expt_idx}")
                print(contingency_table[t][ffn_name][expt_idx])
                table = contingency_table[t][ffn_name][expt_idx]
                num_elements_zero = (table == 0).sum()
                if num_elements_zero >= 3:
                    print(f"No elements for expert {expt_idx} at timestep {t} and ffn {ffn_name}")
                    continue
                # check if elements 
                # calculate the chi square value
                chi2, p, dof, ex = stats.chi2_contingency(contingency_table[t][ffn_name][expt_idx])
                print(f"Chi2 value: {chi2}, p value: {p}")
                # if p value is less than 0.05, then the expert is skilled
                if p < 0.05:
                    print(f"Expert {expt_idx} is skilled at timestep {t} and ffn {ffn_name}")


if __name__ == "__main__":
    main()
