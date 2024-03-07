import json
import os
import sys
import torch
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

model_name = 'runwayml/stable-diffusion-v1-5'
res_path = 'results/stable-diffusion/fine-tuned-relu'
res_path = os.path.join(res_path, model_name)
folder_name = 'moefication'
if not os.path.exists(os.path.join(res_path, folder_name, 'selection_frequency')):
    os.makedirs(os.path.join(res_path, folder_name, 'selection_frequency'))

file = os.path.join(res_path, folder_name, 'expert_counter_0.2.json')
with open(file, 'r') as file:
    data = json.load(file)

timesteps = data.keys()
for t in timesteps:
    print("Saving freq plot for t=", t)
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    layer_names = data[t].keys()
    freq_per_expert = [data[t][name] for name in layer_names]
    # sort the freq per expert
    freq_per_expert = [sorted(freq, reverse=True) for freq in freq_per_expert]
    # plot the freq per expert as a bar plot
    for i, freq in enumerate(freq_per_expert):
        ax[i//4, i%4].bar(range(len(freq)), freq)
        ax[i//4, i%4].set_title(f'Layer {i}')
        ax[i//4, i%4].set_xlabel('Expert index')
        ax[i//4, i%4].set_ylabel('Selection Frequency')
        # draw horizontal line for topk
        ax[i//4, i%4].axhline(y=float(0.2), color='black', linestyle='--')
        # vertical line at topk * num_experts
        ax[i//4, i%4].axvline(x=float(0.2)*len(freq), color='green', linestyle='--')
        
    plt.savefig(os.path.join(res_path, folder_name, 'selection_frequency', f'freq_{t}.png'))
    plt.close()
    plt.title(f'Freq plot for time step {t} with {float(0.2)*100} % experts')
    print("Saved freq plot for t=", t)

# read all the files
# files = os.listdir(os.path.join(res_path, folder_name))
# files = [f for f in files if f.endswith('.json')]
# print(files)
# num_time_steps = 51


#     with open(os.path.join(res_path, folder_name, f), 'r') as file:
#         data = json.load(file)
#         timesteps = data.keys()
#         # select only the timesteps in intervals of 10
#         timesteps = [t for t in timesteps if int(t) % 10 == 0]
#         topk = f.split('.json')[0].split('_')[-1]

#         if not os.path.exists(os.path.join(res_path, folder_name, f.split(".json")[0])):
#             os.makedirs(os.path.join(res_path, folder_name, f.split(".json")[0]))

#         for t in timesteps:
#             print("Saving freq plot for t=", t)
#             fig, ax = plt.subplots(4, 4, figsize=(20, 20))
#             layer_names = data[t].keys()
#             freq_per_expert = [data[t][name] for name in layer_names]
#             # sort the freq per expert
#             freq_per_expert = [sorted(freq, reverse=True) for freq in freq_per_expert]
#             # plot the freq per expert as a bar plot
#             for i, freq in enumerate(freq_per_expert):
#                 ax[i//4, i%4].bar(range(len(freq)), freq)
#                 ax[i//4, i%4].set_title(f'Layer {i}')
#                 ax[i//4, i%4].set_xlabel('Expert index')
#                 ax[i//4, i%4].set_ylabel('Selection Frequency')
#                 # draw horizontal line for topk
#                 ax[i//4, i%4].axhline(y=float(topk), color='black', linestyle='--')
#                 # vertical line at topk * num_experts
#                 ax[i//4, i%4].axvline(x=float(topk)*len(freq), color='green', linestyle='--')
                
#             plt.savefig(os.path.join(res_path, folder_name, f.split(".json")[0], f'freq_{t}.png'))
#             plt.close()
#             plt.title(f'Freq plot for time step {t} with {float(topk)*100} % experts')
#             print("Saved freq plot for t=", t)



