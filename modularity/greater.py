import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.append(os.getcwd())
import utils


# load expert clusters from param split
def load_expert_clusters(files, folder):
    expert_clusters = {}
    for file in files:
        path = os.path.join(folder, 'param_split', file)
        data = torch.load(path)
        expert_clusters[file] = data
    return expert_clusters


def main():
    args = utils.Config('experiments/mod_config_greater.yaml', 'modularity')
    args.configure('modularity')

    # load parameter splits from moefication
    param_split = os.listdir(os.path.join(args.res_path, 'param_split'))

    # load expert clusters
    expert_clusters = load_expert_clusters(param_split, args.res_path)
    param_names = sorted(expert_clusters.keys())

    # get neuron predictivity values
    predictivity_data = {}
    predictivity_data['base'] = json.load(open(os.path.join(args.save_path, args.modularity['condition']['base_prompts'])))
    predictivity_data['adj'] = json.load(open(os.path.join(args.save_path, args.modularity['condition']['concept_prompts'])))

    for t_step in range(0, len(predictivity_data['base']['time_steps'].keys())):
        print(f'Processing timestep {t_step}')
        # for every timestep plot, average predictivity per expert per layer
        fig, ax = plt.subplots(4, 4, figsize=(28, 28))
        fig.suptitle(f'timestep {t_step}', fontsize=30)
        # make horizontal space between subplots
        fig.subplots_adjust(hspace=0.5)
        # make horizontal space between title and subplots
        # plt.subplots_adjust(top=1.5)
        base_prompt_pred = predictivity_data['base']['time_steps'][str(t_step)]
        adj_prompt_pred = predictivity_data['adj']['time_steps'][str(t_step)]
        for i in range(len(base_prompt_pred.keys())):
            base_avg = np.array(base_prompt_pred[str(i)]['avg'])
            base_std = np.array(base_prompt_pred[str(i)]['std'])
            adj_avg = np.array(adj_prompt_pred[str(i)]['avg'])
            adj_std = np.array(adj_prompt_pred[str(i)]['std'])
            is_greater = adj_avg > base_avg + args.modularity['margin']
            
            avg_exp_pred = [] # list will store expert specialisation score for a layer
            # cluster neurons into experts
            num_experts = max(expert_clusters[param_names[i]]) + 1
            for ext_idx in range(num_experts):
                # get neurons that belong to this expert
                neurons = torch.tensor(expert_clusters[param_names[i]]) == ext_idx
                avg_pred_expert = base_avg[neurons].mean()
                avg_pred_expert_adj = adj_avg[neurons].mean()
                print(f'Expert {ext_idx} avg predictivity: {avg_pred_expert} -> {avg_pred_expert_adj}')
                # calculate the average predictivity for this expert
                is_expert_skilled = is_greater[neurons]                 
                avg_exp_pred.append(np.mean(is_expert_skilled) * 100.0)
            
            # save expert indices that are skilled
            skilled_experts = np.where(np.array(avg_exp_pred) > args.modularity['condition']['skill_ratio'] * 100.0)[0].tolist()
            # save as json file
            with open(os.path.join(args.modularity['skill_expert_path'], f'timestep_{t_step}_layer_{i}.json'), 'w') as f:
                json.dump(skilled_experts, f)

            ax[i//4, i%4].plot(avg_exp_pred, marker='o')
            ax[i//4, i%4].set_title(f'Layer {i}', fontsize=20)
            ax[i//4, i%4].set_xlabel('Expert', fontsize=20)
            ax[i//4, i%4].set_ylabel('Percentage of skilled neurons', fontsize=20)
        plt.tight_layout()
        # plt.savefig('test_images/experts_accuracy_greater_50.png')
        plt.savefig(os.path.join(args.modularity['plots'], f'avg_neuron_value_timestep_{t_step}.png'))
        plt.close()


if __name__ == "__main__":
    main()