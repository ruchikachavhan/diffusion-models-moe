import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class WandaRemoveNeurons(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, remove_timesteps=None):
        super(WandaRemoveNeurons, self).__init__(seed, T, n_layers, replace_fn, keep_nsfw)
        self.expert_indices = {}
        for j in range(0, n_layers):
            # read .pt file
            print(os.path.join(path_expert_indx, f'layer_{j}.json'))
            self.expert_indices[j] = torch.load(os.path.join(path_expert_indx, f'layer_{j}.pt'))
            # convert to half because of an error
            self.expert_indices[j] = self.expert_indices[j].half()
            print(f'layer_{j}.json', self.expert_indices[j].sum())
                
        self.timestep = 0
        self.layer = 0
        self.gates = []
        self.replace_fn = replace_fn
        self.remove_timesteps = remove_timesteps

    def hook_fn(self, module, input, output):
        args = (1.0,)

        # Change fc1 with mask 
        old_weights = module.fc1.weight.clone()
        new_weights = old_weights * (1 - self.expert_indices[int(self.layer%12)].to(old_weights.device))
        module.fc1.weight = torch.nn.Parameter(new_weights)
        hidden_states = module.fc1(input[0])

        # Replace the weights with old weights
        module.fc1.weight = torch.nn.Parameter(old_weights)

        # Apply activation function
        hidden_states = module.activation_fn(hidden_states)

        hidden_states = module.fc2(hidden_states)
        self.update_layer()
        return hidden_states
    
        # # get hidden state
        # if self.replace_fn == GEGLU:
        #     # Change the projection matrix, multiply the binary mask with the weights of the projection matrix
        #     # last half of te projection matrix
        #     # if self.timestep < 20:
        #     old_weights = module.proj.weight[module.proj.weight.shape[0]//2:, :].clone()
        #     # pruning
        #     new_weights = old_weights * (1 - self.expert_indices[self.timestep][self.layer].to(old_weights.device))
        #     module.proj.weight[module.proj.weight.shape[0]//2:, :] = new_weights
                            
        #     hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)

        #     # replace the weights with old weights
        #     module.proj.weight[module.proj.weight.shape[0]//2:, :] = old_weights
            
        #     # apply gelu
        #     gate = module.gelu(gate)
                
        #     hidden_states = hidden_states * gate
        #     self.gates.append(gate.detach().cpu())

        # elif self.replace_fn == GELU:
        #     hidden_states = module.proj(input[0])
        #     hidden_states = module.gelu(hidden_states)
        #     expert_indx = self.expert_indices[self.timestep][self.layer]
        #     if len(expert_indx) > 0:
        #         if self.timestep <= 5:
        #             indx = torch.where(torch.tensor(expert_indx) == 1)[0]
        #             hidden_states[:, :, indx] = 0
            
        #     self.gates.append(hidden_states.detach().cpu())

        # self.update_time_layer()

        # return hidden_states
    
    
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