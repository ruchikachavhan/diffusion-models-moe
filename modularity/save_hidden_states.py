import json
import os
import sys
import torch
import tqdm
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import SaveStates


def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
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

        neuron_pred_adj.save(os.path.join(args.modularity['hidden_states_path'], f'hidden_states_adj_{iter}.pth'))
        neuron_pred_base.save(os.path.join(args.modularity['hidden_states_path'], f'hidden_states_base_{iter}.pth'))


if __name__ == "__main__":
    main()