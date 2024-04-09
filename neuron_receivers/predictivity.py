import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
import utils


class NeuronPredictivity(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers, replace_fn = GEGLU, keep_nsfw=False):
        super(NeuronPredictivity, self).__init__(seed, replace_fn, keep_nsfw)
        self.T = T
        self.n_layers = n_layers
        self.predictivity = utils.Average()
        self.max_gate = {}
        self.replace_fn = replace_fn
    
    
    def hook_fn(self, module, input, output):
        # get hidden state
        hidden_states = module.fc1(input[0])
        hidden_states = module.activation_fn(hidden_states)
        max_act = torch.max(hidden_states.view(-1, hidden_states.shape[-1]), dim=0)[0]
        self.max_gate = max_act.detach().cpu().numpy()
        self.predictivity.update(max_act.detach().cpu().numpy())
        hidden_states = module.fc2(hidden_states)
        return hidden_states
        # save the out
        # args = (1.0,)
        # if self.replace_fn == GEGLU:
        #     hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        #     # gate is of the shape (bs, seq len, hidden size). During evaluation batch size is 1
        #     # so we can reshape it to (seq len, hidden size) and take the max activation over entire sequence
        #     max_act = torch.max(module.gelu(gate).view(-1, gate.shape[-1]), dim=0)[0]
        #     self.max_gate[self.timestep][self.layer] = max_act.detach().cpu().numpy()
        #     self.predictivity.update(max_act.detach().cpu().numpy(), self.timestep, self.layer)
        #     self.update_time_layer()
        #     return hidden_states * module.gelu(gate)
        # else:
        #     hidden_states = module.proj(input[0])
        #     hidden_states = module.gelu(hidden_states)
        #     max_act = torch.mean(hidden_states.view(-1, hidden_states.shape[-1]), dim=0)
        #     self.max_gate[self.timestep][self.layer] = max_act.detach().cpu().numpy()
        #     self.predictivity.update(max_act.detach().cpu().numpy(), self.timestep, self.layer)
            
        #     self.update_time_layer()
        #     return hidden_states
        

    def test(self, model, ann = 'A brown dog in the snow'):
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