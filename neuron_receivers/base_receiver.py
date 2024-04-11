import torch
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

class BaseNeuronReceiver:
    '''
    This is the base class for storing and changing activation functions
    '''
    def __init__(self, seed = 0, replace_fn = GEGLU, keep_nsfw = False):
        self.seed = seed
        self.gates = []
        self.hidden_states = []
        self.keep_nsfw = keep_nsfw
        print("Keep nsfw: ", keep_nsfw)
        if self.keep_nsfw:
            print("Removing safety checker")
            safety_checker.StableDiffusionSafetyChecker.forward = sc
        self.safety_checker = safety_checker.StableDiffusionSafetyChecker
        self.replace_fn = replace_fn
        self.remove_token_idx = None
    
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
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, self.replace_fn) and 'mlp' in name and 'encoder.layers' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)
                if bboxes is not None:
                    module.bounding_box = bboxes[name + '.proj.weight']
                else:
                    module.bounding_box = None
        print("Number of hooks: ", num_modules)

        # do same for text_encoder_2
        # for name, module in model.text_encoder_2.named_modules():
        #     if isinstance(module, self.replace_fn) and 'mlp' in name and 'encoder.layers' in name:
        #         # print(name)
        #         hook = module.register_forward_hook(self.hook_fn)
        #         num_modules += 1
        #         hooks.append(hook)
        #         if bboxes is not None:
        #             module.bounding_box = bboxes[name + '.proj.weight']
        #         else:
        #             module.bounding_box = None

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # out = model(ann, safety_checker=self.safety_checker, num_inference_steps=4, guidance_scale=8.0).images[0]
        out = model(ann, safety_checker=self.safety_checker).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.gates
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # Method to write a test case
        raise NotImplementedError
    
