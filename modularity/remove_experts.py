import json
import os
import sys
import torch
import numpy as np
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import NeuronSpecialisation
sys.path.append('moefication')
from helper import modify_ffn_to_experts

def remove_experts(adj_prompts, model, neuron_receiver, args, val=False):
    iter = 0
    for ann_adj in adj_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seeed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # run model for the original text
        out = model(ann_adj).images[0]

        neuron_receiver.reset_time_layer()
        out_adj, pred_adj = neuron_receiver.observe_activation(model, ann_adj)
        # save images
        prefix = args.modularity['remove_expert_path'] if not val else args.modularity['remove_expert_path_val']
        out.save(os.path.join(prefix, f'img_{iter}.jpg'))
        out_adj.save(os.path.join(prefix, f'img_{iter}_adj.jpg'))
        iter += 1

def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    args.configure('modularity')

    # model 
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks
    neuron_receiver = NeuronSpecialisation(seed=args.seed, path_expert_indx = args.modularity['skill_expert_path'], T=args.timesteps, n_layers=num_geglu)
                                           
    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]

    # COnvert FFns into moe
    model, _, _ = modify_ffn_to_experts(model, args)

    if args.fine_tuned_unet is not None:
        neuron_receiver.test(model, relu_condition=args.fine_tuned_unet is not None)
        print("Neuron receiver test passed")
    
    # remove experts
    remove_experts(adj_prompts, model, neuron_receiver, args)

    # read val_dataset
    with open(f'modularity/val_things_{adjectives}.txt') as f:
        val_objects = f.readlines()
    
    val_base_prompts = [f'a {thing.strip()}' for thing in val_objects]
    
    # remove experts from val_dataset
    remove_experts(val_base_prompts, model, neuron_receiver, args, val=True)
    

if __name__ == "__main__":
    main()