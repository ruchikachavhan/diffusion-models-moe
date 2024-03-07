import torch
import json
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
import utils
import cv2

class NeuronPredictivityBB(BaseNeuronReceiver):
    def __init__(self, seed, T, n_layers):
        super(NeuronPredictivityBB, self).__init__(seed)
        self.T = T
        self.n_layers = n_layers
        self.predictivity = utils.StatMeter(T, n_layers)
        
        self.timestep = 0
        self.layer = 0

        # bounding box is a list of indices that are within the bounding box
        self.bounding_box = None
    
    def update_time_layer(self):
        if self.layer == 15:
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
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        # gate is of the shape (bs, seq len, hidden size). During evaluation batch size is 1
        # so we can reshape it to (seq len, hidden size) and take the max activation over entire sequence
        gate = module.gelu(gate)
        # For every latent vector, we choose only the activations that are within the bounding box
        if module.bounding_box is not None:
            gate_within_bb = gate[:, module.bounding_box, :]
        max_act = torch.max(gate_within_bb.view(-1, gate.shape[-1]), dim=0)[0]

        self.predictivity.update(max_act.detach().cpu().numpy(), self.timestep, self.layer)
        self.update_time_layer()
        return hidden_states * gate
    
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # hook the model
        latent_space_seq_length = [4096, 4096, 1024, 1024, 256, 256, 64, 256, 256, 256, 1024, 1024, 1024, 4096, 4096, 4096]
        latent_space_seq_length = [np.sqrt(x) for x in latent_space_seq_length]
        bboxes = json.load(open('test_images/test_bbox.json'))

        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)
                module.bounding_box = bboxes[name + '.proj.weight']

        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test bounding box and image coordinates
        # read bounding box file from test_images
        
        for i, layer_name in enumerate(bboxes.keys()):
            np_out = np.array(out)
            bounding_box = (bboxes[layer_name])
            # scale according to factor 
            factor = 512 // latent_space_seq_length[i]
            # bounding box corrdinates coorespond to flattened vector of latent space of dimension (latent_space_seq_length[i], latent_space_seq_length[i])
            # convert it to 2D coordinates
            bounding_box = [(x // latent_space_seq_length[i], x % latent_space_seq_length[i]) for x in bounding_box]
            top_left = bounding_box[0]
            bottom_right = bounding_box[-1]
            # scale the coordinates
            top_left = (top_left[0] * factor, top_left[1] * factor)
            bottom_right = (bottom_right[0] * factor, bottom_right[1] * factor)            
            
            # convert to int
            top_left = (int(top_left[1]), int(top_left[0]))
            bottom_right = (int(bottom_right[1]), int(bottom_right[0]))

            # convert output to numpy
            # draw the bounding box
           # draw points on img
            cv2.rectangle(np_out, top_left, bottom_right, (0, 255, 0), 2)
            cv2.imwrite(f'test_images/bbox_test_function_{i}.png', np_out)


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