import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from collections import Counter


class GetExperts(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, experts_per_layer, layer_names):
        super(GetExperts, self).__init__(seed)
        self.label_counter = {}
        self.score_tracker = {}
        self.T = T
        self.n_layers = n_layers
        self.experts_per_layer = experts_per_layer
        self.layer_names = layer_names
        for t in range(T):
            self.label_counter[t] = {}
            self.score_tracker[t] = {}
            for i in range(n_layers):
                self.label_counter[t][i] = []
                self.score_tracker[t][i] = []
        
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
        for t in range(self.T):
            self.label_counter[t] = {}
            self.score_tracker[t] = {}
            for i in range(self.n_layers):
                self.label_counter[t][i] = []
                self.score_tracker[t][i] = []
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
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)

            # get tokens that belong to bounding boxes
            if module.bounding_box is not None:
                try:
                    score_within_bb = score.view(bsz, seq_len, -1)[:, module.bounding_box, :]
                    score_within_bb = score_within_bb.view(-1, score_within_bb.shape[-1])
                except:
                    score_within_bb = score.view(-1, score.shape[-1]).clone()
            else:
                score_within_bb = score.view(-1, score.shape[-1]).clone()

            topk_scores = torch.max(score_within_bb, dim=0)[0]
            self.score_tracker[self.timestep][self.layer] = topk_scores.detach().cpu()
            max_activated_expert = torch.max(score_within_bb, dim=1)[1]
            # count the frequency of selecting each expert
            freq = Counter(max_activated_expert.detach().cpu().numpy())
            # sort the keys according to values
            freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
            # select top 50% most frequently selected experts
            max_activated_expert = list(freq.keys())[:int(len(freq) * 0.5)]
            self.label_counter[self.timestep][self.layer] = max_activated_expert


            # select neurons based on the expert labels
            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
        
        self.update_time_layer()
        hidden_states = hidden_states * gate
        return hidden_states
    
    