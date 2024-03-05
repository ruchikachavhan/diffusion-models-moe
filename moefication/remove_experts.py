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
sys.path.append('modularity')
from modularity_analysis import NeuronPredictivity
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
import relufy_model
from diffusers.models.activations import GEGLU
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
sys.path.append('moefication')
from helper import modify_ffn_to_experts


class NeuronSpecialisation(NeuronPredictivity):
    def __init__(self, path_expert_indx, T, n_layers):
        super(NeuronSpecialisation, self).__init__(T, n_layers)
        self.expert_indices = {}
        for i in range(0, T):
            self.expert_indices[i] = {}
            for j in range(0, n_layers):
                # read file 
                self.expert_indices[i][j] = json.load(open(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.json'), 'r'))
                print(f'timestep_{i}_layer_{j}.json', self.expert_indices[i][j])
        self.timestep = 0
        self.layer = 0
        self.gates = []

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
            expert_indx = self.expert_indices[self.timestep][self.layer]
            # make a binary mask of the expert indices
            mask = torch.ones_like(labels)
            # expert_indx is a tensor containing the expert indices
            # choose the expert indices from the labels
            if len(expert_indx) > 0:
                for idx in expert_indx:
                    mask[labels == idx] = 0
            # from labels, remove the elements where mask is 0
            labels = labels[mask == 1]
            labels = labels.view(bsz, seq_len, -1)
            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
        
        self.update_time_layer()
                
        hidden_states = hidden_states * gate
        self.gates.append(gate.detach().cpu())
        return hidden_states
    
    def test(self, model, ann = 'an white cat', relu_condition = False):
        # hook the model
        torch.manual_seed(0)
        np.random.seed(0)
        nochange_out = model(ann).images[0]
        nochange_out.save('test_images/test_image_all_expert.png')
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test if all gates have positive values if relu_condition is True
        for gate in self.gates:
            assert torch.all(gate >= 0) == relu_condition, "All gates should be positive"

        # save test image
        out.save('test_images/test_image_expert_removal.png')
        self.gates = []

def remove_experts(adj_prompts, model, neuron_receiver, args, val=False):
    iter = 0
    for ann_adj in adj_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0) 
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
    args = utils.Config('experiments/config.yaml', 'modularity')
    args.configure('modularity')

    # model 
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks
    condition = args.modularity['condition']
    path = args.modularity[f'skilled_experts_{condition}']
    neuron_receiver = NeuronSpecialisation(path_expert_indx = path, T = args.timesteps, n_layers = args.n_layers)
                                           
    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]

    # COnvert FFns into moe
    model = modify_ffn_to_experts(model, args)

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