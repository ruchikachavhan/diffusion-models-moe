from ast import arg
from re import template
import json
import os
from PIL import Image
import sys
import numpy as np
import torch
import argparse
import tqdm
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
from diffusers.models.activations import GEGLU
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchvision import transforms



def coco_dataset(data_path, split):
    with open(os.path.join(data_path, f'annotations/captions_{split}2014.json')) as f:
        data = json.load(f)
    data = data['annotations']
    # select 30 k images randomly
    np.random.seed(0)
    np.random.shuffle(data)
    data = data[:100]
    imgs = [os.path.join(data_path, f'{split}2014', 'COCO_' + split + '2014_' + str(ann['image_id']).zfill(12) + '.jpg') for ann in data]
    anns = [ann['caption'] for ann in data]
    return imgs, anns

class NeuronReceiver:
    def __init__(self):
        self.gates = []
        self.hidden_states = []
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        self.gates.append(module.gelu(gate).detach().cpu())
        return hidden_states * module.gelu(gate)
    
    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, img, ann):
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

        print(f"Number of modules: {num_modules}")
        # forward pass
        #  fix seed
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann, **SafetyConfig.MEDIUM).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.gates

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../COCO-vqa', help='path to the coco dataset')
    parser.add_argument('--blocks-to-change', nargs='+', help='blocks to change the activation function')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--res-path', type=str, default='results/stable-diffusion/', help='path to store the results of moefication')
    parser.add_argument('--dbg', action='store_true', help='debug mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # make image directory
    if not os.path.exists(os.path.join(args.res_path, 'images')):
        os.makedirs(os.path.join(args.res_path, 'images'))
    
    
    model_id = "runwayml/stable-diffusion-v1-5"
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(args.gpu)

    imgs, anns = coco_dataset(args.data_path, 'val')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    neuron_receiver = NeuronReceiver()
    iter = 0
    all_results = {}
    all_results['time_steps'] = {}
    for t in range(51):
        all_results['time_steps'][t] = {}
        all_results['time_steps'][t]['min'] = [0.0 for _ in range(16)]
        all_results['time_steps'][t]['max'] = [0.0 for _ in range(16)]
        all_results['time_steps'][t]['neg_ratio'] = [0.0 for _ in range(16)]
        all_results['time_steps'][t]['zero_ratio'] = [0.0 for _ in range(16)]

    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        # if iter > 100:
        #     break
        print("Iter: ", iter)
        print("text: ", ann)
        out, gates = neuron_receiver.observe_activation(model, img, ann)
        # divide gate into chunks of 51
        # print the sparsity ratio of each chunk
        for i in range(0, len(gates), 16):
            gate_timestep = gates[i:i+16]
            for j, gate in enumerate(gate_timestep):
                if j > 15:
                    continue
                # check sparsity
                # check if values of the gate are < -2.5
                # if yes, then it is a sparse neuron
                # calculate ratio of sparse neurons per layer per time step
                # print the ratio
                # min value of gate
                
                all_results['time_steps'][i//16]['min'][j] = min(all_results['time_steps'][i//16]['min'][j], gate.min().item())
                # max value of gate
                all_results['time_steps'][i//16]['max'][j] = max(all_results['time_steps'][i//16]['max'][j], gate.max().item())

                # check for negative values
                mask = gate < 0.0
                neg_ratio = mask.int().sum(-1).float() / gate.shape[-1]
                neg_ratio = neg_ratio.mean()
                all_results['time_steps'][i//16]['neg_ratio'][j] += neg_ratio.item()

                # check for values between -1e-3 and 1e-3
                mask = (gate > -1e-3) & (gate < 1e-3)
                zero_ratio = mask.int().sum(-1).float() / gate.shape[-1]
                zero_ratio = zero_ratio.mean()
                all_results['time_steps'][i//16]['zero_ratio'][j] += zero_ratio.item()

        iter += 1

    # calculate the mean of the ratios
    for t in range(51):
        for j in range(16):
            all_results['time_steps'][t]['neg_ratio'][j] /= iter
            all_results['time_steps'][t]['zero_ratio'][j] /= iter
    print(all_results)
    with open(os.path.join(args.res_path, 'sparsity_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Results saved at: ", os.path.join(args.res_path, 'sparsity_results.json'))

        
    
if __name__ == '__main__':
    main()

