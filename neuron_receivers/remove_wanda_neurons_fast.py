import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity
from diffusers.models.lora import LoRACompatibleLinear
import pickle

class WandaRemoveNeuronsFast(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, remove_timesteps=51, weights_shape=None):
        super(WandaRemoveNeuronsFast, self).__init__(seed, T, n_layers, replace_fn, keep_nsfw)
        self.expert_indices = {}
        for i in range(0, T):
            self.expert_indices[i] = {}
            for j in range(0, n_layers):
                # print(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.pkl'))
                with open(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.pkl'), 'rb') as f:
                    # load sparse matrix from pickle file
                    indices = pickle.load(f)
                    # convert to array
                    self.expert_indices[i][j] = torch.tensor(indices.toarray())
                
        self.timestep = 0
        self.layer = 0
        self.gates = []
        self.replace_fn = replace_fn
        self.remove_timesteps = remove_timesteps

    def hook_fn(self, module, input, output):
        args = (1.0,)

        # get hidden state
        if self.replace_fn == GEGLU:
            # Change the projection matrix, multiply the binary mask with the weights of the projection matrix
            # last half of te projection matrix
            old_weights = module.proj.weight[module.proj.weight.shape[0]//2:, :].clone()
            # pruning
            new_weights = old_weights * (1 - self.expert_indices[self.timestep][self.layer].to(old_weights.device))
            module.proj.weight[module.proj.weight.shape[0]//2:, :] = new_weights
                            
            hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)

            # replace the weights with old weights
            module.proj.weight[module.proj.weight.shape[0]//2:, :] = old_weights
            
            # apply gelu
            gate = module.gelu(gate)
                
            hidden_states = hidden_states * gate
            self.gates.append(gate.detach().cpu())

        elif self.replace_fn == GELU:
            hidden_states = module.proj(input[0])
            hidden_states = module.gelu(hidden_states)
            expert_indx = self.expert_indices[self.timestep][self.layer]
            if len(expert_indx) > 0:
                if self.timestep <= 5:
                    indx = torch.where(torch.tensor(expert_indx) == 1)[0]
                    hidden_states[:, :, indx] = 0
            
            self.gates.append(hidden_states.detach().cpu())

        self.update_time_layer()

        return hidden_states
    
    def linear_hook_fn(self, module, input, output):
        # Linear (lora compatible layer)
        # change wieghts by applying the binary mask
        old_weights = module.weight.clone()

        if self.timestep < self.remove_timesteps and self.remove_timesteps is not None:
            # read the expert indices
            binary_mask = self.expert_indices[self.timestep][self.layer]
            binary_mask = binary_mask.to(old_weights.device)
            new_weights = old_weights * (1 - binary_mask)

            # Apply the forward pass with matrix multiplication
            # output_dim, input_dim = new_weights.shape
            # implement matrix multiplication in torch.nn.Linear 
            hidden_states = torch.nn.functional.linear(input[0], new_weights, module.bias)

            # copy the weights into proj
            # proj.weight = torch.nn.Parameter(new_weights)
            # proj.bias = torch.nn.Parameter(module.bias)
            # hidden_states = proj(input[0])
            # assert hidden_states.shape == output.shape, "Output shape should be same as hidden states"
        else:
            # use old weights
            hidden_states = torch.nn.functional.linear(input[0], old_weights, module.bias)
        
        # replace the weights with old weights
        self.update_time_layer()
        return hidden_states


    def observe_activation(self, model, ann, bboxes=None):
        # hook the model
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                hook = module.register_forward_hook(self.linear_hook_fn)
                num_modules += 1
                hooks.append(hook)

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # to account for batches
        if isinstance(ann, list):
            out = model(ann).images
        else:
            out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.gates

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