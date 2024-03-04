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
from relufy_model import BaseNeuronReceiver
from diffusers.models.activations import GEGLU


class FrequencyMeasure(BaseNeuronReceiver):
    def __init__(self):
        super(FrequencyMeasure, self).__init__()
        self.label_counter = []
    
    def hook_fn(self, module, input, output):
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        gate = module.gelu(gate)

        if module.patterns is not None:
            k = module.k
            bsz, seq_len, hidden_size = gate.shape
            gate_gelu = gate.clone()
            gate_gelu = gate_gelu.view(-1, hidden_size)
            score = torch.matmul(gate_gelu, module.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            # update counter for which expert was selected for this layer
            self.label_counter.append(labels[0, :, :].detach().cpu().numpy())
            # select neurons based on the expert labels
            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
                
        hidden_states = hidden_states * gate
        return hidden_states
    
    def clear_counter(self):
        self.label_counter = []
    
    
def main():
    args = utils.Config('experiments/config.yaml', 'moefication')
    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication['topk_experts'] = topk
        args.configure('moefication')
    print(f"Topk experts: {args.moefication['topk_experts']}")

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Change FFNS to a mixture of experts
    model = modify_ffn_to_experts(model, args)

    # Eval dataset
    imgs, anns = utils.coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    # Initialise expert counter
    expert_counter, ffn_names_list = initialise_expert_counter(model)
    neuron_receiver = FrequencyMeasure()

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 5 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0)
        # With MOEfication
        neuron_receiver.clear_counter()
        out_moe, _ = neuron_receiver.observe_activation(model, ann)
        label_counter = neuron_receiver.label_counter
        
        iter += 1

        for i in range(0, len(label_counter), num_geglu):
            gate_timestep = label_counter[i:i+num_geglu]
            for j, labels in enumerate(gate_timestep):
                if j > num_geglu:
                    continue
                for label in labels:
                    expert_counter[i//num_geglu][ffn_names_list[j]][label] += (1.0 / len(labels))
    
    # divide by number of images
    for t in range(args.timesteps):
        for ffn_name in ffn_names_list:
            expert_counter[t][ffn_name] /= iter
            expert_counter[t][ffn_name] = expert_counter[t][ffn_name].tolist()

    # save the expert counter
    topk_experts = args.moefication['topk_experts']
    with open(os.path.join(args.save_path, f'expert_counter_{topk_experts}.json'), 'w') as f:
        json.dump(expert_counter, f)
        

if __name__ == "__main__":
    main()