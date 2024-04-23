import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class RemoveNeuronsNoiseHPO(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, dof_val, conf_val, trials_per_layer=None, replace_fn = GEGLU, keep_nsfw=False):
        super(RemoveNeuronsNoiseHPO, self).__init__(seed, T, n_layers, replace_fn, keep_nsfw)
        self.expert_indices = {}
        for i in range(0, T):
            self.expert_indices[i] = {}
            for j in range(0, n_layers):
                # read file 
                dof_conf_folder = f"dof_{dof_val}_conf_{conf_val}"
                print(os.path.join(path_expert_indx, dof_conf_folder, f'timestep_{i}_layer_{j}.json'))
                self.expert_indices[i][j] = json.load(open(os.path.join(path_expert_indx, dof_conf_folder, f'timestep_{i}_layer_{j}.json'), 'r'))
                # print(f'timestep_{i}_layer_{j}.json', self.expert_indices[i][j])
        self.timestep = 0
        self.layer = 0
        self.gates = []
        self.replace_fn = replace_fn
        self.trial = trials_per_layer
        self.update_for_t = {}
        self.unet_output = {}
        for t in range(T):
            self.unet_output[t] = []
    
    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
        # reset the gate 
        for t in range(self.T):
            self.max_gate[t] = {}
            self.unet_output[t] = []
            for l in range(self.n_layers):
                self.max_gate[t][l] = []

    def hook_fn(self, module, input, output):
        args = (1.0,)

        # get hidden state
        if self.replace_fn == GEGLU:
            hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
            # apply gelu
            gate = module.gelu(gate)

            expert_indx = self.expert_indices[self.timestep][self.layer]
            # if self.layer == 0:
            #         # decide if you want to remove the expert neurons for that timestep 
            #         self.update_for_t[self.timestep] = self.trial.suggest_categorical(f"timestep_{self.timestep}", [0, 1])
            # # else:/
            # if self.update_for_t[self.timestep] == 1:
                # remove those neurons from gate
            if len(expert_indx) > 0:
                indx = torch.where(torch.tensor(expert_indx) == 1)[0]
                gate[:, :, indx] = -0.17

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

    def unet_hook_fn(self, module, input, output):
        # because of the way timestep is updated, it is updated before this forward hook is executed, hence time-step - 1
        self.unet_output[self.timestep-1] = output[0].detach().cpu()
        return output
    
    def observe_activation(self, model, ann, bboxes=None):
        hooks = []
        # reset the gates
        self.gates = []

        # hook the MLPs of the model
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, self.replace_fn) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)
                if bboxes is not None:
                    module.bounding_box = bboxes[name + '.proj.weight']
                else:
                    module.bounding_box = None
        
        # add hook to unet to get noise    
        unet_hook = model.unet.register_forward_hook(self.unet_hook_fn)

        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # out = model(ann, safety_checker=self.safety_checker, num_inference_steps=4, guidance_scale=8.0).images[0]
        out = model(ann, safety_checker=self.safety_checker).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        self.remove_hooks([unet_hook])
        return out, self.unet_output
    
    
    
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