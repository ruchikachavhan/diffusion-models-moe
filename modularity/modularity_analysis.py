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
from relufy_model import BaseNeuronReceiver
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from diffusers.models.activations import GEGLU
from sklearn.metrics import average_precision_score
from sklearn import preprocessing


class NeuronPredictivity(BaseNeuronReceiver):
    def __init__(self, T, n_layers):
        super(NeuronPredictivity, self).__init__()
        self.T = T
        self.n_layers = n_layers
        self.predictivity = utils.StatMeter(T, n_layers)
        
        self.timestep = 0
        self.layer = 0
    
    def update_time_layer(self):
        if self.layer == 15:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1

    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
    
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        # gate is of the shape (bs, seq len, hidden size). During evaluation batch size is 1
        # so we can reshape it to (seq len, hidden size) and take the max activation over entire sequence
        max_act = torch.max(module.gelu(gate).view(-1, gate.shape[-1]), dim=0)[0]

        self.predictivity.update(max_act.detach().cpu().numpy(), self.timestep, self.layer)
        self.update_time_layer()
        return hidden_states * module.gelu(gate)
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # hook the model
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
        
        # test if all gates have positive values
        for t in range(self.timestep):
            for l in range(self.layer):
                gate = self.predictivity.results['time_steps'][t][l]['avg'].avg
                std = self.predictivity.results['time_steps'][t][l]['std'].stddev()
                assert torch.all(gate > 0), f"Relu failed, max activation is expected to be positive"

        # save test image
        out.save('test_images/test_image_mod.jpg')
        
        # reset the predictivity
        self.predictivity = utils.StatMeter(self.T, self.n_layers)

def main():
    args = utils.Config('experiments/config.yaml', 'modularity')
    args.configure('modularity')

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks to measure predictivity
    neuron_pred_base = NeuronPredictivity(T=args.timesteps, n_layers=num_geglu)
    neuron_pred_adj = NeuronPredictivity(T=args.timesteps, n_layers=num_geglu)

    # Test the model
    if args.fine_tuned_unet is not None:
        neuron_pred_base.test(model)
        neuron_pred_adj.test(model)
        print("Neuron receiver tests passed")

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

        # save images
        out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
        out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
        iter += 1

    # save results
    print("Saving results")
    neuron_pred_adj.predictivity.save(os.path.join(args.save_path, 'predictivity_adj.json'))
    neuron_pred_base.predictivity.save(os.path.join(args.save_path, 'predictivity_base.json'))



if __name__ == "__main__":
    main()
