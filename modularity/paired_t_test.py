import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.append(os.getcwd())
import utils
from greater import load_expert_clusters
import scipy.stats as stats

def main():
    args = utils.Config('experiments/mod_config_t_test.yaml', 'modularity')
    args.configure('modularity')

    param_split = os.listdir(os.path.join(args.res_path, 'param_split'))

    # load expert clusters
    expert_clusters = load_expert_clusters(param_split, args.res_path)
    param_names = sorted(expert_clusters.keys())

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    base_prompts = [f'a {thing.strip()}' for thing in objects]
    # add an adjective of choice to every element in things list
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]

    # Initialise average and standard deviation for base and adj prompts
    base_avg, adj_avg, diff_std = {}, {}, {}
    for t in range(0, args.timesteps):
        base_avg[t], adj_avg[t], diff_std[t] = {}, {}, {}
        for l in range(0, args.n_layers):
            base_avg[(t, l)] = utils.Average()
            adj_avg[(t, l)] = utils.Average()
            diff_std[(t, l)] = utils.StandardDev()

    # For each time step and layer, calculate the average neuron activation of the base and adj prompts
    # Also calculate standard deviation of the difference between base and adj prompts over all prompts
    for i in range(1, len(base_prompts)+1):
        print(f"Processing prompt {i}")
        base_states = torch.load(os.path.join(args.modularity['hidden_states_path'], f'hidden_states_base_{i}.pth'))
        adj_states = torch.load(os.path.join(args.modularity['hidden_states_path'], f'hidden_states_adj_{i}.pth'))
        for t in range(args.timesteps):
            for l in range(args.n_layers):
                feat_dim = base_states[t][l].shape[-1]
                base_states_ = base_states[t][l].reshape(-1, feat_dim).max(0)[0]
                adj_states_ = adj_states[t][l].reshape(-1, feat_dim).max(0)[0]
                base_avg[(t, l)].update(base_states_)
                adj_avg[(t, l)].update(adj_states_)

                diff = base_states_ - adj_states_
                diff_std[(t, l)].update(diff)

    # Perform the t test 
    skilled_neurons = {}
    for t in range(args.timesteps):
        skilled_neurons[t] = {}
        for l in range(args.n_layers):
            base = base_avg[(t, l)].avg
            adj = adj_avg[(t, l)].avg
            diff = diff_std[(t, l)].stddev() + 1e-6
            # t value
            t_value = (base - adj) / (diff /  len(base_prompts) ** 0.5)
            t_value = t_value.cpu().numpy()
            # one sided upper tail t-test with alpha = 0.05
            # If neuron passes the t-test, it is considered skilled
            skilled_neurons[t][l] = torch.tensor(t_value < -1.645)

            # check which expert the skilled neurons belong to
            expert_indices = expert_clusters[param_names[l]]
            skilled_experts = []
            for expt_idx in range(max(expert_indices) + 1):
                # get the skilled neurons in the expert
                neurons_in_expert = torch.tensor(expert_clusters[param_names[l]]) == expt_idx
                # if ratio of skilled neurons to total neurons in the expert is greater than 0.5
                ratio = skilled_neurons[t][l][neurons_in_expert].sum() / neurons_in_expert.sum()
                print(f"Ratio of skilled neurons in expert {expt_idx}: {ratio}")
                if ratio > 0.4:
                    skilled_experts.append(expt_idx)
            
            print(f"Skilled experts at time step {t} and layer {l}: {len(skilled_experts)}")
            # for every time step and layer, we will save the skilled experts
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(skilled_experts, f)

if __name__ == "__main__":
    main()
