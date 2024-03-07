import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

# class to save hidden states 
class SaveStates(NeuronPredictivity):
    def __init__(self, seed, T, n_layers):
        super(SaveStates, self).__init__(seed, T, n_layers)
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
