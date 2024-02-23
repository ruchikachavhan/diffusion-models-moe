import json
import os
import sys
import torch
import tqdm
import types
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
import moe_utils
sys.path.append(os.getcwd())
import utils as dm_utils
import eval_coco as ec
from diffusers.models.activations import GEGLU
from eval_moefied_sd import modify_ffn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../COCO-vqa', help='path to the coco dataset')
    parser.add_argument('--blocks-to-change', nargs='+', default=['down_block', 'mid_block', 'up_block'], help='blocks to change the activation function')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--res-path', type=str, default='results/stable-diffusion/', help='path to store the results of moefication')
    parser.add_argument('--dbg', action='store_true', help='debug mode')
    parser.add_argument('--num-images', type=int, default=1000, help='number of images to test')
    parser.add_argument('--fine-tuned-unet', type = str, default = None, help = "path to fine-tuned unet model")
    parser.add_argument('--model-id', type=str, default="runwayml/stable-diffusion-v1-5", help='model id')
    parser.add_argument('--timesteps', type=int, default=51, help='number of denoising time steps')
    parser.add_argument('--num-layer', type=int, default=3, help='number of layers')
    parser.add_argument('--topk-experts', type=float, default=1, help='ratio of experts to select')
    parser.add_argument('--templates', type=str, 
                        default='{}.{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight',
                        help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

    args = parser.parse_args()
    return args

class FrequencyMeasure:
    def __init__(self):
        self.label_counter = []
    
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
            # update counter for which expert was selected for this layer
            self.label_counter.append(labels[0, :, :].detach().cpu().numpy())

            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
                
        hidden_states = hidden_states * gate
        return hidden_states
    
    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, img, ann):
        hooks = []
        # reset the gates
        self.label_counter = []

        # hook the model
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        print(f"Number of modules: {num_modules}")
        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.label_counter
    
def main():
    args = get_args()
    dm_utils.make_dirs(args)
  
    model, num_geglu = dm_utils.get_sd_model(args)
    model = model.to(args.gpu)

    expert_counter = {}
    # make a expert counter for every time step
    for i in range(args.timesteps):
        expert_counter[i] = {}

    ffn_names_list = []
    # Modify FFN to add expert labels
    for name, module in model.unet.named_modules():
        if 'ff.net' in name and isinstance(module, GEGLU):
            ffn_name = name + '.proj.weight'
            path = os.path.join(args.res_path, 'param_split', ffn_name)
            modify_ffn(module, path, args.topk_experts)
            for t in range(args.timesteps):
                expert_counter[t][ffn_name] = np.array([0.0 for _ in range(module.patterns.shape[0])])
            ffn_names_list.append(ffn_name)

    imgs, anns = dm_utils.coco_dataset(args.data_path, 'val', args.num_images)
    neuron_receiver = FrequencyMeasure()

    # sort the ffn names list
    ffn_names_list.sort()

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 10 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0)
        # With MOEfication
        out_moe, label_counter = neuron_receiver.observe_activation(model, img, ann)
        
        iter += 1

        for i in range(0, len(label_counter), num_geglu):
            gate_timestep = label_counter[i:i+num_geglu]
            for j, labels in enumerate(gate_timestep):
                if j > num_geglu:
                    continue
                for label in labels:
                    expert_counter[i//num_geglu][ffn_names_list[j]][label] += (1.0 / len(labels))
    
    # divide by number of images
    for t in range(args.timesteps):
        for ffn_name in ffn_names_list:
            expert_counter[t][ffn_name] /= iter
            expert_counter[t][ffn_name] = expert_counter[t][ffn_name].tolist()

    print(expert_counter)
    # save the expert counter
    with open(os.path.join(args.res_path, args.model_id, 'moefication', f'expert_counter_{args.topk_experts}.json'), 'w') as f:
        json.dump(expert_counter, f)
        

if __name__ == "__main__":
    main()