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
from mod_utils import get_prompts

def main():
    args = utils.Config('experiments/mod_config_t_test.yaml', 'modularity')
    args.configure('modularity')

    param_split = os.listdir(os.path.join(args.res_path, 'param_split'))

    # load expert clusters
    expert_clusters = load_expert_clusters(param_split, args.res_path)
    param_names = sorted(expert_clusters.keys())

    # Dataset from things.txt
    # read things.txt

    base_prompts, _, _ = get_prompts(args)

    # read predicitivity files 
    predictivity_data = {}
    predictivity_data['base'] = json.load(open(os.path.join(args.save_path, args.modularity['condition']['base_prompts'])))
    predictivity_data['adj'] = json.load(open(os.path.join(args.save_path, args.modularity['condition']['concept_prompts'])))
    # read standard dev path
    diff_std = json.load(open(os.path.join(args.save_path, args.modularity['condition']['diff_std'])))

    # Perform the t test 
    skilled_neurons = {}
    for t in range(args.timesteps):
        skilled_neurons[t] = {}
        for l in range(args.n_layers):
            print(f"Processing time step {t} and layer {l}")
            base = np.array(predictivity_data['base']['time_steps'][str(t)][str(l)]['avg'])
            adj = np.array(predictivity_data['adj']['time_steps'][str(t)][str(l)]['avg'])
            diff = np.array(diff_std[str(t)][str(l)])
            # t value
            t_value = (base - adj) / (diff /  len(base_prompts) ** 0.5)
            # one sided upper tail t-test with alpha = 0.05
            # If neuron passes the t-test, it is considered skilled
            skilled_neurons[t][l] = torch.tensor(t_value < -1.769)

            # check which expert the skilled neurons belong to
            expert_indices = expert_clusters[param_names[l]]
            skilled_experts = []
            for expt_idx in range(max(expert_indices) + 1):
                # get the skilled neurons in the expert
                neurons_in_expert = torch.tensor(expert_clusters[param_names[l]]) == expt_idx
                # if ratio of skilled neurons to total neurons in the expert is greater than 0.5
                ratio = skilled_neurons[t][l][neurons_in_expert].sum() / neurons_in_expert.sum()
                # print(f"Ratio of skilled neurons in expert {expt_idx}: {ratio}")
                if ratio > args.modularity['condition']['skill_ratio']:
                    skilled_experts.append(expt_idx)
            
            print(f"Skilled experts at time step {t} and layer {l}: {len(skilled_experts)}")
            # for every time step and layer, we will save the skilled experts
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(skilled_experts, f)

            # save skilled neurons indices
            # convert to list and int
            skilled_neurons[t][l] = [int(i) for i in skilled_neurons[t][l]]
            with open(os.path.join(args.modularity['skill_neuron_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(skilled_neurons[t][l], f)

if __name__ == "__main__":
    main()
