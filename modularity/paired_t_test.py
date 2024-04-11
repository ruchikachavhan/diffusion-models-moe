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

    # param_split = os.listdir(os.path.join(args.res_path, 'param_split'))

    # load expert clusters
    # expert_clusters = load_expert_clusters(param_split, args.res_path)
    # param_names = sorted(expert_clusters.keys())

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
    # for t in range(args.timesteps):
        # skilled_neurons[t] = {}
    for l in range(args.n_layers):
        print(f"Processing layer {l}")
        base = np.array(predictivity_data['base'][str(l)])
        adj = np.array(predictivity_data['adj'][str(l)])
        diff = np.array(diff_std[str(l)])
        # std_base = np.array(predictivity_data['base']['time_steps'][str(t)][str(l)]['std'])
        # std_adj = np.array(predictivity_data['adj']['time_steps'][str(t)][str(l)]['std'])
                    # t value
        t_value = (base - adj) / ( diff /  (len(base_prompts) ** 0.5))
        # print(f"t_value: {diff}")
        # one sided upper tail t-test with alpha = 0.05
        # If neuron passes the t-test, it is considered skilled

        # bloody red - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -2.624)

        # nudity - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -2.567)

        # painting styles - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -2.539)

        # Van Gogh style, Monet, Pablo Picasso - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -2.539)

        # manga - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -3.883)

        # gender - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -3.169)

        # scene removal - SD
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.959)

        # bloody red - LCM
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.408)

        # naked - LCM
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.408)

        # painting style - LCM, Van Gogh, Monet, 
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.959)

        # Picasso, manga - LCM
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.408)

        # gender - LCM
        # skilled_neurons[t][l] = torch.tensor(t_value < -2.365)

        # scene removal - LCM
        # skilled_neurons[t][l] = torch.tensor(t_value < -5.959)

        # Van Gogh - Text encoder - SD
        # replace inf or nan with 0
        t_value = np.nan_to_num(t_value, nan=0, posinf=0, neginf=0)
        skilled_neurons[l] = torch.tensor(t_value < -1.729)
        
        
        print(f"Skilled neurons at and layer {l}: {skilled_neurons[l].sum()}")
        # skilled_neurons[t][l] = torch.tensor(abs(t_value) > 2.02)



        # check which expert the skilled neurons belong to
        # expert_indices = expert_clusters[param_names[l]]
        # skilled_experts = []
        # for expt_idx in range(max(expert_indices) + 1):
        #     # get the skilled neurons in the expert
        #     neurons_in_expert = torch.tensor(expert_clusters[param_names[l]]) == expt_idx
        #     # if ratio of skilled neurons to total neurons in the expert is greater than 0.5
        #     ratio = skilled_neurons[t][l][neurons_in_expert].sum() / neurons_in_expert.sum()
        #     # print(f"Ratio of skilled neurons in expert {expt_idx}: {ratio}")
        #     if ratio > args.modularity['condition']['skill_ratio']:
        #         skilled_experts.append(expt_idx)
        
        # print(f"Skilled experts at time step {t} and layer {l}: {len(skilled_experts)}")
        # # for every time step and layer, we will save the skilled experts
        # with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
        #     json.dump(skilled_experts, f)

        # save skilled neurons indices
        # convert to list and int
        skilled_neurons[l] = [int(i) for i in skilled_neurons[l]]
        with open(os.path.join(args.modularity['skill_neuron_path'], f'layer_{l}.json'), 'w') as f:
            json.dump(skilled_neurons[l], f)

if __name__ == "__main__":
    main()
