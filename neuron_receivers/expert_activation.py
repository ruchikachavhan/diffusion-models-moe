import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from collections import Counter
import utils


class ExpertPredictivity(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, keep_nsfw=False):
        super(ExpertPredictivity, self).__init__(seed, keep_nsfw)
        self.T = T
        self.n_layers = n_layers
        self.predictivity = utils.StatMeter(T, n_layers)
        self.keep_nsfw = keep_nsfw
        self.max_gate = {}
        for t in range(T):
            self.max_gate[t] = {}
            for i in range(n_layers):
                self.max_gate[t][i] = []
        
        # initialise timestep and layer id
        self.timestep = 0
        self.layer = 0
        self.sample_id = 0
    
    def update_time_layer(self):
        if self.layer == 15:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1
    
    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
    
    def reset(self):
        self.max_gate = {}
        for t in range(self.T):
            self.max_gate[t] = {}
            for i in range(self.n_layers):
                self.max_gate[t][i] = []
        self.reset_time_layer()

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
            expert_act = torch.max(score, dim=0)[0]
            self.max_gate[self.timestep][self.layer] = expert_act.detach().cpu().numpy()
            self.predictivity.update(expert_act.detach().cpu().numpy(), self.timestep, self.layer)
        
        self.update_time_layer()
        hidden_states = hidden_states * gate
        return hidden_states
    
    def test(self, model):
        return True
        
    