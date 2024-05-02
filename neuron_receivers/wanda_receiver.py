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
        self.predictivity = utils.TimeLayerColumnNorm(T, n_layers)
        self.timestep = 0
        self.layer = 0
        self.replace_fn = replace_fn
    
    def update_time_layer(self):
        if self.layer == self.n_layers - 1:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1

    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        if self.replace_fn == GEGLU:
            hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
            ''' 
            Store the norm of the gate for each layer and timestep in the predictivity object 
            '''
            # save_gate = input[0].view(-1, input[0].shape[-1]).detach().cpu()
            # save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)

            out = hidden_states * module.gelu(gate)
            save_gate = out.view(-1, out.shape[-1]).detach().cpu()
            # # take max over the sequence length
            # save_gate = torch.max(save_gate, dim=0)[0]
            save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
            self.predictivity.update(save_gate, self.timestep, self.layer)

            self.update_time_layer()

            return hidden_states * module.gelu(gate)
    
    def test(self, model, ann = ['a cat', 'a dog']):
        '''Test if the norm calculator works okay'''

        # Register the hook
        # hook the model
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, self.replace_fn) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out1 = model(ann[0]).images[0]
        # remove the hook
        self.remove_hooks(hooks)

        self.reset_time_layer()
        # hook the model again
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, self.replace_fn) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out2 = model(ann[1]).images[0]
        
        incremental_norms = self.predictivity.get_column_norms()
        # test if all gates have positive values
        for t in range(self.timestep):
            for l in range(self.layer):
                total_matrix = self.predictivity[t][t].A
                norm = np.linalg.norm(total_matrix)

                # incremental norm 
                calc_norm = incremental_norms[t][l]
                assert np.isclose(norm, calc_norm), f'Norms do not match: {norm} != {calc_norm}'
        print("Norms match")
        return True