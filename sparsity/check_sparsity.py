import json
import os
import sys
import torch
import yaml
import tqdm
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
from relufy_model import BaseNeuronReceiver
from diffusers.models.activations import GEGLU
sys.path.append(os.getcwd())
from utils import get_sd_model, coco_dataset, Config, StatMeter


class SparsityMeasure(BaseNeuronReceiver):
    def __init__(self):
        super(SparsityMeasure, self).__init__()

    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        self.gates.append(module.gelu(gate).detach().cpu())
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
        for gate in self.gates:
            assert torch.all(gate >= 0), f"Relu failed"

        # save test image
        out.save('test_images/test_image_relu.jpg')
        self.gates = []


def main():
    args = Config('experiments/config.yaml', 'sparsity')
    # Model
    model, num_geglu = get_sd_model(args)
    model = model.to(args.gpu)
    # Eval dataset
    imgs, anns = coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    # Neuron receiver to store gates for every sample
    neuron_receiver = SparsityMeasure()
    if args.fine_tuned_unet is not None:
        neuron_receiver.test(model)
        print("Neuron receiver test passed")

    iter = 0
    results = StatMeter(T=args.timesteps, n_layers=num_geglu)
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 5 and args.dbg:
            break
        print("Iter: ", iter)
        print("text: ", ann)
        out, gates = neuron_receiver.observe_activation(model, ann)
        
        # divide gate into chunks of number of time steps
        for i in range(0, len(gates), num_geglu):
            gate_timestep = gates[i:i+num_geglu]
            for j, gate in enumerate(gate_timestep):
                if j > num_geglu:
                    continue
                # check sparsity
                # check if values of the gate == 0
                mask = gate == 0.0
                # % of neurons that are 0 out of total neurons (= hidden dimension)
                exact_zero_ratio = mask.int().sum(-1).float() / gate.shape[-1]
                # Take mean over all tokens
                exact_zero_ratio = exact_zero_ratio.mean()
                results.update(exact_zero_ratio.item(), i//num_geglu, j)
        iter += 1

    print(f'Saving results to {args.save_path}')
    results.save(os.path.join(args.save_path, 'sparsity.json'))

        
    
if __name__ == '__main__':
    main()

