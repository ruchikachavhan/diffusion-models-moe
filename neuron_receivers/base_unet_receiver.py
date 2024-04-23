import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class BaseUNetReceiver(NeuronPredictivity):
    '''
    class to only store the output of the unet
    '''
    def __init__(self, seed, T, n_layers,):
        super(BaseUNetReceiver, self).__init__(seed, T, n_layers, GEGLU, keep_nsfw=False)
        self.unet_output = {}
        for t in range(T):
            self.unet_output[t] = []
    
    def update_timestep(self):
        self.timestep += 1
    
    def reset_time(self):
        self.timestep = 0
        for t in range(self.T):
            self.unet_output[t] = []

    def unet_hook_fn(self, module, input, output):
        self.unet_output[self.timestep] = output[0].detach().cpu()
        # update timestep
        self.update_timestep()
        return output

    def observe_activation(self, model, ann, bboxes=None):
        
        # add hook to unet to get noise    
        unet_hook = model.unet.register_forward_hook(self.unet_hook_fn)

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # out = model(ann, safety_checker=self.safety_checker, num_inference_steps=4, guidance_scale=8.0).images[0]
        out = model(ann, safety_checker=self.safety_checker).images[0]

        # remove the hook
        self.remove_hooks([unet_hook])
        return out, self.unet_output 