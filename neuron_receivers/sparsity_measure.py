import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver

class SparsityMeasure(BaseNeuronReceiver):
    '''
    Measure sparsity of the model
    '''
    def __init__(self, seed):
        super(SparsityMeasure, self).__init__(seed)

    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        self.gates.append(module.gelu(gate).detach().cpu())
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
        for gate in self.gates:
            assert torch.all(gate >= 0), f"Relu failed"

        # save test image
        out.save('test_images/test_image_relu.jpg')
        self.gates = []