import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
import utils
from torch.nn.functional import relu


class Wanda(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, replace_fn = GEGLU, keep_nsfw=False):
        super(Wanda, self).__init__(seed, replace_fn, keep_nsfw)
        self.T = T
        self.n_layers = n_layers
        self.predictivity = {}
        for l in range(self.n_layers):
            self.predictivity[l] = utils.ColumnNormCalculator()
        self.timestep = 0
        self.layer = 0
        self.replace_fn = replace_fn
    
    def update_layer(self):
        self.layer += 1

    def reset_layer(self):
        self.layer = 0 
    
    def hook_fn(self, module, input, output):
        hidden_states = module.fc1(input[0])
        '''
        Store the norm of the input for each layer
        '''
        save_gate = input[0].clone().detach().cpu()
        save_gate = save_gate.view(-1, input[0].shape[-1])
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        if self.layer < self.n_layers:
            self.predictivity[self.layer].add_rows(save_gate)
        
        hidden_states = module.activation_fn(hidden_states)
        hidden_states = module.fc2(hidden_states)
        self.update_layer()
        return hidden_states
    