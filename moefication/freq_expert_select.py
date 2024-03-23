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
import moe_utils
from helper import modify_ffn_to_experts, initialise_expert_counter
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import FrequencyMeasure
    
def main():
    args = utils.Config('experiments/moefy_config.yaml', 'moefication')
    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication['topk_experts'] = topk
        args.configure('moefication')
    print(f"Topk experts: {args.moefication['topk_experts']}")

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Change FFNS to a mixture of experts
    model, ffn_names_list, num_experts_per_ffn = modify_ffn_to_experts(model, args)

    # Eval dataset
    imgs, anns = utils.coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    neuron_receiver = FrequencyMeasure(args.seed, args.timesteps, args.n_layers, num_experts_per_ffn, ffn_names_list)

    iter = 0
    
    # bad solution for averaging but had to do it
    expert_counter = {}
    for t in range(args.timesteps):
        expert_counter[t] = {}
        for ffn_name in ffn_names_list:
            expert_counter[t][ffn_name] = [0] * num_experts_per_ffn[ffn_name]

    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        # if iter > 5 and args.dbg:
        #     break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)    
        neuron_receiver.reset()
        out_moe, _ = neuron_receiver.observe_activation(model, ann)
        label_counter = neuron_receiver.label_counter

        # add to expert counter
        for t in range(args.timesteps):
            for ffn_name in ffn_names_list:
                for expert in range(num_experts_per_ffn[ffn_name]):
                    expert_counter[t][ffn_name][expert] += label_counter[t][ffn_names_list.index(ffn_name)][expert] / args.inference['num_images']
        
        iter += 1

    # print(expert_counter)
    # # save the expert counter
    topk_experts = args.moefication['topk_experts']
    with open(os.path.join(args.save_path, f'expert_counter_{topk_experts}.json'), 'w') as f:
        json.dump(expert_counter, f)
        

if __name__ == "__main__":
    main()