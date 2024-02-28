import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import json


model_name = 'runwayml/stable-diffusion-v1-5'
res_path = 'results/stable-diffusion/fine-tuned-relu/sparsity'
num_act = 16

# read all the files
files = os.listdir(os.path.join(res_path, model_name))
# get the sparsity files
sparsity_files = [f for f in files if 'sparsity' in f]

num_samples = 1000
num_timesteps = 50
neg_values = [[] for _ in range(num_timesteps)]
pos_values = [[] for _ in range(num_timesteps)]
exact_zero_values = [[] for _ in range(num_timesteps)]
zero_values = [[] for _ in range(num_timesteps)]

for f in sparsity_files:
    with open(os.path.join(res_path, model_name, f), 'r') as file:
        data = json.load(file)
        for t in range(num_timesteps):
            exact_zero_values[t].append(data['time_steps'][str(t)]['exact_zero_ratio'])


neg_values = np.array(neg_values)
pos_values = np.array(pos_values)
exact_zero_values = np.array(exact_zero_values)
zero_values = np.array(zero_values)

print(neg_values.shape, pos_values.shape, exact_zero_values.shape, zero_values.shape)

# plot the sparsity for every time step
for t in range(num_timesteps):
    # plot a box plot for every layer
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=exact_zero_values[t]*100.0)
    plt.title(f'Sparsity for time step {t}')
    plt.xlabel('Layer')
    plt.ylabel('Percentage of sparse neurons')
    plt.legend()
    # plt.tight_layout()
    # plt.xticks([i for i in range(0, 70, 10)], [f'{i}' for i in range(0, 70, 10)])
    plt.savefig(os.path.join(res_path, model_name, 'plots', f'sparsity_{t}.png'))
