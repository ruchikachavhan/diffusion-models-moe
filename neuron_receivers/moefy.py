import torch
import numpy as np
from diffusers.models.activations import GEGLU
from neuron_receivers.base_receiver import BaseNeuronReceiver

class MOEFy(BaseNeuronReceiver):
    def __init__(self):
        super(MOEFy, self).__init__()

    def hook_fn(self, module, input, output):
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        gate = module.gelu(gate)

        if module.patterns is not None:
            k = module.k
            bsz, seq_len, hidden_size = gate.shape
            gate_gelu = gate.clone()
            gate_gelu = gate_gelu.view(-1, hidden_size)
            score = torch.matmul(gate_gelu, module.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
                
        self.gates.append(gate.detach().cpu())
        hidden_states = hidden_states * gate
        return hidden_states
    
    def test(self, model, ann = 'A brown dog in the snow', relu_condition = False):
        # hook the model
        torch.manual_seed(0)
        np.random.seed(0)
        nochange_out = model(ann).images[0]
        nochange_out.save('test_images/nochange_out.png')
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
        # test if all gates have positive values if relu_condition is True
        for gate in self.gates:
            assert torch.all(gate >= 0) == relu_condition, "All gates should be positive"
        # save test image
        out.save('test_images/test_image_moe.png')
        self.gates = []
        
