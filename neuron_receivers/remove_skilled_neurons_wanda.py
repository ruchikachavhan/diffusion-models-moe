import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity
import pickle

class WandaRemoveNeurons(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, remove_timesteps=None):
        super(WandaRemoveNeurons, self).__init__(seed, T, n_layers, replace_fn, keep_nsfw)
        self.expert_indices = {}
        for j in range(0, n_layers):
            # read .pt file
            print(os.path.join(path_expert_indx, f'layer_{j}.pkl'))
            with open(os.path.join(path_expert_indx, f'layer_{j}.pkl'), 'rb') as f:
                indices = pickle.load(f)
                self.expert_indices[j] = indices.toarray()
                
        self.timestep = 0
        self.layer = 0
        self.gates = []
        self.replace_fn = replace_fn
        self.remove_timesteps = remove_timesteps

    def hook_fn(self, module, input, output):

        # Change fc1 with mask 
        old_weights = module.fc2.weight.clone()
        binary_mask = torch.tensor(self.expert_indices[int(self.layer%self.n_layers)]).to(old_weights.device)
        new_weights = old_weights * (1 - binary_mask)

        output_dim, input_dim = new_weights.shape
        proj = torch.nn.Linear(input_dim, output_dim)
        proj.weight = torch.nn.Parameter(new_weights)
        proj.bias = module.fc2.bias


        hidden_states = module.fc1(input[0])       
        hidden_states = module.activation_fn(hidden_states)
        # Pass through modified fc2 weights
        hidden_states = proj(hidden_states)

        assert hidden_states.shape == output.shape, f"Hidden states shape {hidden_states.shape} should be equal to output shape {output.shape}"

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