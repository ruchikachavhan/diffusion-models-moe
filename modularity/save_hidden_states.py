import json
import os
import sys
import torch
import tqdm
import types
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from modularity_analysis import NeuronPredictivity


# class to save hidden states 
class SaveStates(NeuronPredictivity):
    def __init__(self, T, n_layers):
        super(SaveStates, self).__init__(T, n_layers)
        self.T = T
        self.n_layers = n_layers
        self.hidden_states = {}
        for t in range(T):
            self.hidden_states[t] = {}
            for l in range(n_layers):
                # For every layer we will save the hidden states
                self.hidden_states[t][l] = torch.empty(0)
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        gate = module.gelu(gate)
        self.hidden_states[self.timestep][self.layer] = gate.detach().cpu()

        # update the time and layer
        self.update_time_layer()

        return hidden_states * gate

    def save(self, path):
        torch.save(self.hidden_states, path)

def main():
    args = utils.Config('experiments/config.yaml', 'modularity')
    args.configure('modularity')

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks to measure predictivity
    neuron_pred_base = SaveStates(T=args.timesteps, n_layers=num_geglu)
    neuron_pred_adj = SaveStates(T=args.timesteps, n_layers=num_geglu)

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    base_prompts = [f'a {thing.strip()}' for thing in objects]
    # add an adjective of choice to every element in things list
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]

    iter = 0
    for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
        if iter >= 5 and args.dbg:
            break
        print("text: ", ann, ann_adj)
        
        neuron_pred_base.reset_time_layer()
        out, _ = neuron_pred_base.observe_activation(model, ann)

        neuron_pred_adj.reset_time_layer()
        out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj)
    
        iter += 1

        neuron_pred_adj.save(os.path.join(args.save_path, f'hidden_states_adj_{iter}.pth'))
        neuron_pred_base.save(os.path.join(args.save_path, f'hidden_states_base_{iter}.pth'))


if __name__ == "__main__":
    main()