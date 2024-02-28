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
sys.path.append(os.getcwd())
import utils as dm_utils
import eval_coco as ec
import relufy_model
from diffusers.models.activations import GEGLU
from sklearn.metrics import average_precision_score
from sklearn import preprocessing


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
    parser.add_argument('--adjective', type=str, default='white', help='adjective to add to things')
    args = parser.parse_args()
    return args

class NeuronPredictivity:
    def __init__(self):
        self.predictivity = []
        self.hidden_states = []
    
    def hook_fn(self, module, input, output):
        # save the out
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        max_act = torch.max(module.gelu(gate).view(-1, gate.shape[-1]), dim=0).values
        self.predictivity.append(max_act.detach().cpu().unsqueeze(0))
        return hidden_states * module.gelu(gate)
    
    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, ann):
        hooks = []
        # reset the gates
        self.predictivity = []

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
        return out, self.predictivity
    
    def test(self, model, ann = 'A brown dog in the snow'):
        # hook the model
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        print(f"Number of modules: {num_modules}")
        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test if all gates have positive values
        for gate in self.predictivity:
            assert torch.all(gate >= 0), f"Relu failed"

        # save test image
        out.save('test_image_relu.jpg')
        
        self.predictivity = []


def get_ap(predictivity1, predictivity2, inv_ap = False):
    # Calculate mean average precision for every neuron 
    # for every time step
    # for every sample
    print("Length of predictivity: ", len(predictivity1), len(predictivity2))
    results = []

    for i in range(0, len(predictivity1)):
        # calculate AP 
        pred1 = predictivity1[i]
        pred2 = predictivity2[i]
        if inv_ap:
            pred1, pred2 = pred2, pred1
        ap = [average_precision_score([0]*pred1.shape[0] + [1]*pred2.shape[0], torch.cat([pred1[:, k], pred2[:, k]]).numpy()) for k in range(pred1.shape[1])]
        results.append(ap)
    print("Results: ", results)
    return results



def main():
    args = get_args()
    dm_utils.make_dirs(args)
    # Neuron receiver with forward hooks
    neuron_receiver = NeuronPredictivity()

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    objects = [f'a {thing.strip()}' for thing in objects]

    # add an adjective of choice to every element in things list
    adj_objects = [f'a {args.adjective} {thing}' for thing in objects]

    # Model 
    model, num_geglu = dm_utils.get_sd_model(args)
    model = model.to(args.gpu)
    if args.fine_tuned_unet is not None:
        neuron_receiver.test(model)
        print("Neuron receiver test passed")

    iter = 0
    all_predictivity, all_predictivity_adj = [torch.empty(0) for _ in range(num_geglu*51)], [torch.empty(0) for _ in range(num_geglu*51)]
    for ann, ann_adj in tqdm.tqdm(zip(objects, adj_objects)):
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0)
        
        # observe the activation
        out, pred = neuron_receiver.observe_activation(model, ann)
        out_adj, pred_adj = neuron_receiver.observe_activation(model, ann_adj)

        all_predictivity = [torch.cat((all_predictivity[i], p)) for i, p in enumerate(pred)]
        all_predictivity_adj = [torch.cat((all_predictivity_adj[i], p)) for i, p in enumerate(pred_adj)]
        iter += 1

    # Calculate the average precision
    results = get_ap(all_predictivity, all_predictivity_adj)
    print(len(results))

    # save results 
    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity'))
    
    with open(os.path.join(args.res_path, args.model_id, 'modularity', 'predictivity.json'), 'w') as f:
        json.dump(results, f)
    
    results_inv = get_ap(all_predictivity, all_predictivity_adj, inv_ap = True)
    with open(os.path.join(args.res_path, args.model_id, 'modularity', 'predictivity_inv.json'), 'w') as f:
        json.dump(results_inv, f)
    


        




if __name__ == "__main__":
    main()
