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
        data = json.load(open(os.path.join(root, 'dof_'+str(dof[c])+'_conf_'+str(conf), 'remove_neurons', 'llava_results.json'), 'r'))
        try:
            style_score_after = data['adj_style_score']
        except:
            style_score_after = data['after_removal']['style_score']
        print(c, conf, style_score_after)
        llava_results[c].append(style_score_after)

# plot llava results
fig, axes = plt.subplots(1, len(concept), figsize=(5*len(concept), 5))
fig.suptitle('Confidence interval vs LLAVA style score after removal')
conf_vals = [100 * (1-c/2) for c in conf_vals]
for i, c in enumerate(concept):
    axes[i].plot(conf_vals, llava_results[c], label='LLAVA')
    # mark the points with circles
    axes[i].scatter(conf_vals, llava_results[c], marker='o', color='red')
    axes[i].set_xlabel('Confidence interval (%)')
    axes[i].set_ylabel('Style score')
    axes[i].set_title(c)
    # axes[i].set_xticks(conf_vals)
    axes[i].set_ylim(0.8, 1.2)
plt.savefig('llava_results.png')


# plt.plot(conf_vals, llava_results, label='LLAVA')
# plt.xlabel('Confidence interval')
# plt.ylabel('Style score')