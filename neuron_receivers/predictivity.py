import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
import utils


class NeuronPredictivity(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, keep_nsfw=False):
        super(NeuronPredictivity, self).__init__(seed, keep_nsfw)
        self.T = T
        self.n_layers = n_layers
        self.predictivity = utils.StatMeter(T, n_layers)
        self.max_gate = {}
        for t in range(T):
            self.max_gate[t] = {}
            for l in range(n_layers):
                self.max_gate[t][l] = []
        
        self.timestep = 0
        self.layer = 0
    
    def update_time_layer(self):
        if self.layer == 15:
            self.layer = 0
            self.timestep += 1
        else:
            self.layer += 1

    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
        # reset the gate 
        for t in range(self.T):
            self.max_gate[t] = {}
            for l in range(self.n_layers):
                self.max_gate[t][l] = []
    
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        # gate is of the shape (bs, seq len, hidden size). During evaluation batch size is 1
        # so we can reshape it to (seq len, hidden size) and take the max activation over entire sequence
        max_act = torch.max(module.gelu(gate).view(-1, gate.shape[-1]), dim=0)[0]
        self.max_gate[self.timestep][self.layer] = max_act.detach().cpu().numpy()

        self.predictivity.update(max_act.detach().cpu().numpy(), self.timestep, self.layer)
        self.update_time_layer()
        return hidden_states * module.gelu(gate)
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # hook the model
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test if all gates have positive values
        for t in range(self.timestep):
            for l in range(self.layer):
                gate = self.predictivity.results['time_steps'][t][l]['avg'].avg
                std = self.predictivity.results['time_steps'][t][l]['std'].stddev()
                assert torch.all(gate > 0), f"Relu failed, max activation is expected to be positive"

        # save test image
        out.save('test_images/test_image_mod.jpg')
        
        # reset the predictivity
        self.predictivity = utils.StatMeter(self.T, self.n_layers)
        # reset gates
        self.reset_time_layer()