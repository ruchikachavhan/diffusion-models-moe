import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver
from neuron_receivers.predictivity import NeuronPredictivity

class AddExperts(NeuronPredictivity):
    def __init__(self, seed, path_expert_indx, T, n_layers, keep_nsfw=False):
        super(AddExperts, self).__init__(seed, T, n_layers, keep_nsfw)
        # read predictivity file 
        adj = path_expert_indx.split('/')[-3]
        base_path = path_expert_indx.split(adj)[0]
        activation_data = json.load(open(os.path.join(base_path, adj, 'predictivity_base_expert.json'), 'r'))
        
        self.expert_indices = {}
        self.avg_activation = {}
       
        for i in range(0, T):
            self.expert_indices[i] = {}
            self.avg_activation[i] = {}
            for j in range(0, n_layers):
                # read file 
                print(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.json'))
                self.expert_indices[i][j] = json.load(open(os.path.join(path_expert_indx, f'timestep_{i}_layer_{j}.json'), 'r'))
                print(f'timestep_{i}_layer_{j}.json', self.expert_indices[i][j])
                
                self.avg_activation[i][j] = activation_data['time_steps'][str(i)][str(j)]['std']
                print(f'avg_activation_{i}_{j}.json', self.avg_activation[i][j])
        self.timestep = 0
        self.layer = 0
        self.gates = []

    def hook_fn(self, module, input, output):
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        gate = module.gelu(gate)

        expert_indx = self.expert_indices[self.timestep][self.layer]
        # if len(expert_indx) > 0:
        #         patterns = module.patterns.clone()
        #         expert_indx = [i for i in range(0, module.patterns.shape[0]) if i not in expert_indx]
        #         patterns[expert_indx, :] = torch.tensor(self.avg_activation[self.timestep][self.layer]).to(patterns.device).to(gate.dtype)
        # else:
        patterns = module.patterns.clone()

        if module.patterns is not None:
            k = module.k
            bsz, seq_len, hidden_size = gate.shape
            gate_gelu = gate.clone()
            gate_gelu = gate_gelu.view(-1, hidden_size)
            score = torch.matmul(gate_gelu, patterns.transpose(0, 1))
            
            score[:, expert_indx] = score[:, expert_indx] + 5.0 * torch.tensor(self.avg_activation[self.timestep][self.layer]).to(gate.device).to(gate.dtype)[expert_indx]

            labels = torch.topk(score, k=int(0.8*k), dim=-1)[1].view(bsz, seq_len, int(0.8*k))
            # print(labels.shape, k)
            # add expert indices to the labels
            # labels = torch.cat((labels, torch.tensor(expert_indx).long().to(labels.device).unsqueeze(0).unsqueeze(0).repeat(bsz, seq_len, 1)), dim=-1)
            cur_mask = torch.nn.functional.embedding(labels, patterns).sum(-2)
            gate[cur_mask == 0] = 0

            # set the values of neurons given by expert_indices to the average activation value
        # gate[:, :, torch.tensor(expert_indx).to(gate.device)] = gate[:, :, torch.tensor(expert_indx).to(gate.device)] + 2.0 * torch.tensor(self.avg_activation[self.timestep][self.layer]).to(gate.device).to(gate.dtype)[torch.tensor(expert_indx).to(gate.device)]

        self.update_time_layer()

        hidden_states = hidden_states * gate
        self.gates.append(gate.detach().cpu())
        return hidden_states
    
    
    def test(self, model, ann = 'an white cat', relu_condition = False):
        # hook the model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        nochange_out = model(ann).images[0]
        nochange_out.save('test_images/test_image_all_expert.png')
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
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