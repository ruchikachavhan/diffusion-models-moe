import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class RemoveNeurons(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, replace_fn = GEGLU, keep_nsfw=False):
        super(RemoveNeurons, self).__init__(seed, T, n_layers, replace_fn, keep_nsfw)
        self.expert_indices = {}
        # for j in range(0, n_layers):
        #     # read file 
        #     print(os.path.join(path_expert_indx, f'layer_{j}.json'))
        #     self.expert_indices[j] = json.load(open(os.path.join(path_expert_indx, f'layer_{j}.json'), 'r'))
        #     print(f'layer_{j}.json', self.expert_indices[j])
        self.layer = 0
        self.gates = []
        self.replace_fn = replace_fn

    def hook_fn(self, module, input, output):
        hidden_states = module.fc1(input[0])
        hidden_states = module.activation_fn(hidden_states)

        # indx = torch.tensor(self.remove_token_idx).to(hidden_states.device)
        # if self.layer < 7:
        indx = torch.tensor([2,3,4])
        hidden_states[:, indx, :] = 0
    
        hidden_states = module.fc2(hidden_states)
        self.update_layer()
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
            if isinstance(module, self.replace_fn) and 'ff.net' in name:
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