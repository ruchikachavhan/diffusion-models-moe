# plot confidence values and scores 
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os

concept = ['naked', 'bloody red', 'Van Gogh']

dof = {}
dof['naked'] = 17
dof['bloody red'] = 14
dof['Van Gogh'] = 19
# read llava results
llava_results = {}
conf_vals = [0.20, 0.10, 0.05, 0.02, 0.01, 0.001]


for c in concept:
    llava_results[c] = []
    for conf in conf_vals:
        root = os.path.join('/raid/s2265822/diffusion-models-moe/results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity', c, 'skilled_neuron_t_test', '0.3', )
        # load text file
        data = open(os.path.join(root, 'dof_'+str(dof[c])+'_conf_'+str(conf), 'remove_neurons', 'fid_score.txt'), 'r').readlines()
        score = data[0].split("(")[1].split(")")[0]
        score = float(score)
        print(c, conf, score)
        llava_results[c].append(score)

# plot llava results
fig, axes = plt.subplots(1, len(concept), figsize=(5*len(concept), 5))
fig.suptitle('Confidence interval vs FID after removal')
conf_vals = [100 * (1-c/2) for c in conf_vals]
for i, c in enumerate(concept):
    axes[i].plot(conf_vals, llava_results[c], label='FID')
    # mark the points with circles
    axes[i].scatter(conf_vals, llava_results[c], marker='o', color='red')
    axes[i].set_xlabel('Confidence interval (%)')
    axes[i].set_ylabel('FID after removal')
    axes[i].set_title(c)
    # axes[i].set_xticks(conf_vals)
    # axes[i].set_ylim(0.8, 1.2)
plt.savefig('fid_results.png')
