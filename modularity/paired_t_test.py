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
import pandas as pd

def critical_value_ranges():
    # Based on the number of objects, we will decide a range of critical values for the t-test
    # read dof file
    dof_critical_values = pd.read_csv('modularity/dof_critical_values.csv')
    # first column in dof and rest columns are confidence intervals
    dof = dof_critical_values['DOF'].values
    conf_int = dof_critical_values.drop('DOF', axis=1)
    # get frst row of confidence intervals
    conf_int = conf_int.keys().values
    conf_int = [float(i) for i in conf_int]
    dof_critical_values_dict = {}

    # access every row of the dataframe
    for index, row in dof_critical_values.iterrows():
        # get the dof
        dof_val = int(row['DOF'])
        critical_values = [row[str(conf)] for conf in conf_int]
        dof_critical_values_dict[str(dof_val)] = {}
        for conf in conf_int:
            dof_critical_values_dict[str(dof_val)][str(conf)] = critical_values[conf_int.index(conf)]
         
    return dof, conf_int, dof_critical_values_dict


def main():
    args = utils.Config('experiments/mod_config_t_test.yaml', 'modularity')
    args.configure('modularity')

    dof, conf_int, dof_critical_values = critical_value_ranges()
    print(dof_critical_values) 

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
    for t in range(args.timesteps):
        skilled_neurons[t] = {}
        for l in range(args.n_layers):
            print(f"Processing time step {t} and layer {l}")
            base = np.array(predictivity_data['base']['time_steps'][str(t)][str(l)]['avg'])
            adj = np.array(predictivity_data['adj']['time_steps'][str(t)][str(l)]['avg'])
            
            diff = np.array(diff_std[str(t)][str(l)])
            std_base = np.array(predictivity_data['base']['time_steps'][str(t)][str(l)]['std'])
            std_adj = np.array(predictivity_data['adj']['time_steps'][str(t)][str(l)]['std'])
                        # t value
            t_value = (base - adj) / (diff /  (len(base_prompts) ** 0.5))

            # for every dof and critical value, we calculate skilled neurons

            for dof_val in dof:
                for conf_val in conf_int:
                    critical_val = dof_critical_values[str(dof_val)][str(conf_val)]
                    # Negative for upper tail test
                    skilled_neurons = torch.tensor(t_value < -critical_val)
                    print(f"Skilled neurons at time step {t} and layer {l} for DOF {dof_val} and confidence interval {conf_val}: {skilled_neurons.sum()} out of {len(skilled_neurons)}")
                    skilled_neurons = [int(i) for i in skilled_neurons]
                    # save in folder 
                    # make a folder for every dof and confidence interval
                    folder_name = f"dof_{dof_val}_conf_{conf_val}"
                    if not os.path.exists(os.path.join(args.modularity['skill_neuron_path'], folder_name)):
                        os.makedirs(os.path.join(args.modularity['skill_neuron_path'], folder_name))
                    with open(os.path.join(args.modularity['skill_neuron_path'], folder_name, f'timestep_{t}_layer_{l}.json'), 'w') as f:
                        json.dump(skilled_neurons, f)

                # one sided upper tail t-test with alpha = 0.05
                # If neuron passes the t-test, it is considered skilled
                # skilled_neurons[t][l] = torch.tensor(t_value < dof_critical_values[str(conf_val)].values[dof_val])

                # print(f"Skilled neurons at time step {t} and layer {l} for dof {dof_val} and confidence interval {conf_val}: {skilled_neurons[t][l].sum()}") 
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

            # Van Gogh - Pixart
            # skilled_neurons[t][l] = torch.tensor(t_value < -2.539)
            
           
            # print(f"Skilled neurons at time step {t} and layer {l}: {skilled_neurons[t][l].sum()}")
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
            # skilled_neurons[t][l] = [int(i) for i in skilled_neurons[t][l]]
            # with open(os.path.join(args.modularity['skill_neuron_path'], f'timestep_{t}_layer_{l}.json'), 'w') as f:
            #     json.dump(skilled_neurons[t][l], f)

if __name__ == "__main__":
    main()
