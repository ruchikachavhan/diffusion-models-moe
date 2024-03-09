import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class NeuronSpecialisation(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers):
        super(NeuronSpecialisation, self).__init__(seed, T, n_layers)
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
            # if module.bounding_box is not None:
            # #     # set gate values and cur_mask values to 0 where cur_mask is False
            #     curr_mask_within_bb  = torch.nn.functional.embedding(labels[:, module.bounding_box, :], module.patterns).sum(-2)
            #     gate[:, module.bounding_box, :][curr_mask_within_bb == 0] = 0
            # else:
            gate[cur_mask == 0] = 0
        
        self.update_time_layer()
                
        hidden_states = hidden_states * gate
        self.gates.append(gate.detach().cpu())
        return hidden_states
    
    
    def test(self, model, ann = 'an white cat', relu_condition = False):
        # hook the model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test if all gates have positive values if relu_condition is True
        for gate in self.gates:
            assert torch.all(gate >= 0) == relu_condition, "All gates should be positive"

        # save test image
        out.save('test_images/test_image_expert_removal.png')
        self.gates = []