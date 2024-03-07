import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import json

# TODO  - Change this code to use the new config file
model_name = 'runwayml/stable-diffusion-v1-5'
res_path = 'results_seed_1/stable-diffusion/fine-tuned-relu'
res_path = os.path.join(res_path, model_name)
num_act = 16

# read the file
with open(os.path.join(res_path, 'sparsity', 'sparsity.json'), 'r') as file:
    data = json.load(file)

for t in data['time_steps'].keys():
    print(f"Processing timestep {t}")
    # plot the sparsity for every layer
    layer_data = data['time_steps'][t] 
    # make a plot for every time step
    plt.figure(figsize=(10, 5))
    averages, stds = [], []
    for layer_id in layer_data.keys():
        # plot a bar plot with standard deviation
        avg = layer_data[layer_id]['avg'] * 100.0
        std = layer_data[layer_id]['std'] * 100.0
        averages.append(avg)
        stds.append(std)

    plt.bar([i for i in range(num_act)], averages, yerr=stds, capsize=5)
    plt.title(f'Sparsity for time step {t}')
    plt.xlabel('Layer ID')
    plt.ylabel('Percentage of sparse neurons')
    # join the points with a black line
    # draw horizontal line at 30% sparsity
    plt.axhline(y=30, color='green', linestyle='--')
    # plt.legend()
    plt.tight_layout()
    plt.xticks([i for i in range(0, num_act, 2)], [f'{i}' for i in range(0, num_act, 2)])
    plt.savefig(os.path.join(res_path, 'sparsity', f'sparsity_{t}.png'))

        





# num_samples = 1000
# num_timesteps = 50
# neg_values = [[] for _ in range(num_timesteps)]
# pos_values = [[] for _ in range(num_timesteps)]
# exact_zero_values = [[] for _ in range(num_timesteps)]
# zero_values = [[] for _ in range(num_timesteps)]

# for f in sparsity_files:
#     with open(os.path.join(res_path, model_name, f), 'r') as file:
#         data = json.load(file)
#         for t in range(num_timesteps):
#             exact_zero_values[t].append(data['time_steps'][str(t)]['exact_zero_ratio'])


# neg_values = np.array(neg_values)
# pos_values = np.array(pos_values)
# exact_zero_values = np.array(exact_zero_values)
# zero_values = np.array(zero_values)

# print(neg_values.shape, pos_values.shape, exact_zero_values.shape, zero_values.shape)

# # plot the sparsity for every time step
# for t in range(num_timesteps):
#     # plot a box plot for every layer
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(data=exact_zero_values[t]*100.0)
#     plt.title(f'Sparsity for time step {t}')
#     plt.xlabel('Layer')
#     plt.ylabel('Percentage of sparse neurons')
#     plt.legend()
#     # plt.tight_layout()
#     # plt.xticks([i for i in range(0, 70, 10)], [f'{i}' for i in range(0, 70, 10)])
#     plt.savefig(os.path.join(res_path, model_name, 'plots', f'sparsity_{t}.png'))
