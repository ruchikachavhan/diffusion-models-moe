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
    assert args.modularity['condition']['name'] == 't_test_expert', "This script is only for t_test_expert condition"
    if args.modularity['condition']['name'] == 't_test_expert':
        fname = args.modularity['condition']['base_prompts']
        args.modularity['condition']['base_prompts'] = fname.split('.json')[0] + '_expert.json'
        fname = args.modularity['condition']['concept_prompts']
        args.modularity['condition']['concept_prompts'] = fname.split('.json')[0] + '_expert.json'
        fname = args.modularity['condition']['diff_std']
        args.modularity['condition']['diff_std'] = fname.split('.json')[0] + '_expert.json'

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
            std_base = np.array(predictivity_data['base']['time_steps'][str(t)][str(l)]['std'])
            std_adj = np.array(predictivity_data['adj']['time_steps'][str(t)][str(l)]['std'])
            diff = np.array(diff_std[str(t)][str(l)])
            # t value
            t_value = (base - adj) / (diff /  len(base_prompts) ** 0.5)
            skilled_experts = torch.tensor(t_value < -3.579)
            # skilled_experts = torch.tensor(t_value < -1.729)
            
            skilled_experts = torch.where(skilled_experts)[0]
            print(f"Skilled experts at time step {t} and layer {l}: {len(skilled_experts)}")

            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(skilled_experts.tolist(), f)

if __name__ == "__main__":
    main()
