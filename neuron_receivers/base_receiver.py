import torch
import numpy as np
from diffusers.models.activations import GEGLU

class BaseNeuronReceiver:
    '''
    This is the base class for storing and changing activation functions
    '''
    def __init__(self):
        self.gates = []
        self.hidden_states = []
    
    def hook_fn(self, module, input, output):
        # custom hook function
        raise NotImplementedError

    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, ann, bboxes=None):
        hooks = []
        # reset the gates
        self.gates = []

        # hook the model
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)
                if bboxes is not None:
                    module.bounding_box = bboxes[name + '.proj.weight']

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.gates
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # Method to write a test case
        raise NotImplementedError
    
