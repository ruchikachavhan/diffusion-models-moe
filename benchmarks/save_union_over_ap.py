import torch
import numpy as np
import scipy
import os
import sys
import pickle
import json

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker


weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]

test_prompts = {
    'memorize': 'Ann Graham Lotz',
    'memorize_0': 'Ann Graham Lotz',
    'memorize_1': 'The No Limits Business Woman Podcast',
    'memorize_2': 'The No Limits Business Woman Podcast',
    'memorize_3': 'The No Limits Business Woman Podcast',
    'memorize_4': 'The No Limits Business Woman Podcast',
    'memorize_5': 'The No Limits Business Woman Podcast',
    'memorize_6': 'The No Limits Business Woman Podcast',
    'memorize_7': 'The No Limits Business Woman Podcast',
    'memorize_8': 'The No Limits Business Woman Podcast',
    'memorize_9': 'The No Limits Business Woman Podcast',
    'memorize_10': 'The No Limits Business Woman Podcast',
    'memorize_11': 'The No Limits Business Woman Podcast',
    'memorize_12': 'The No Limits Business Woman Podcast',
    'memorize_13': 'The No Limits Business Woman Podcast',
    'memorize_14': 'The No Limits Business Woman Podcast',
    'memorize_15': 'The No Limits Business Woman Podcast',
    'memorize_16': 'The No Limits Business Woman Podcast',
    'memorize_17': 'The No Limits Business Woman Podcast',
    'memorize_18': 'The No Limits Business Woman Podcast',
    'memorize_19': 'The No Limits Business Woman Podcast',
}

wanda_thr = {
    'memorize': 0.01,
    'memorize_0': 0.01,
    'memorize_1': 0.01,
    'memorize_2': 0.01,
    'memorize_3': 0.01,
    'memorize_4': 0.01,
    'memorize_5': 0.01,
    'memorize_6': 0.01,
    'memorize_7': 0.01,
    'memorize_8': 0.01,
    'memorize_9': 0.01,
    'memorize_10': 0.01,
    'memorize_11': 0.01,
    'memorize_12': 0.01,
    'memorize_13': 0.01,
    'memorize_14': 0.01,
    'memorize_15': 0.01,
    'memorize_16': 0.01,
    'memorize_17': 0.01,
    'memorize_18': 0.01,
    'memorize_19': 0.01,
}

# Read parameters of SD model and apply the mask
from diffusers import UNet2DConditionModel, StableDiffusionPipeline       
from diffusers.models.activations import GEGLU

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=51)
    parser.add_argument('--n_layers', type=int, default=16)
    parser.add_argument('--concept', type=str, default=None)
    parser.add_argument('--select_ratio', type=float, default=0.0)
    parser.add_argument('--unskilled', action='store_false')
    return parser.parse_args()

def main():

    args = get_args()
    concept = args.concept
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to('cuda:6')

    # generate test image
    gen_seed = 1
    torch.manual_seed(gen_seed)
    np.random.seed(gen_seed)
    prev_image = model(test_prompts[concept], safety_checker=safety_checker_).images[0]
    prev_image.save(f'test_images/{concept}_{gen_seed}_prev.png')

    # get names of layers
    layer_names = []
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
            layer_names.append(name)
            print(module.weight.data.shape)
    # sort 
    layer_names = sorted(layer_names)
    print("Layer names: ", layer_names)

    timesteps = args.timesteps
    n_layers = 16
    seed = 0

    select_ratio = args.select_ratio
    print("Select ratio: ", select_ratio)

    root = f'results/results_seed_%s/stable-diffusion/baseline/{args.model_id}/modularity/%s' 

    path = os.path.join(root % (seed, concept), 'skilled_neuron_AP', str(wanda_thr[concept]))
    if not args.unskilled:
        print("Path before: ", path)
        print("------------------------- WARNING - Experiment wil run by removing unskilled neurons ------------------------------")
        path = path.replace('skilled_neuron_wanda', 'unskilled_neuron_wanda')
        print("Path: ", path)

    union_concepts = {}
    for l in range(n_layers):
        union_concepts[layer_names[l]] = np.zeros(weights_shape[l][-1])
        for t in range(0, args.timesteps):
            fname = os.path.join(path, f'predictivity_{t}_{l}.json')
            with open(fname, 'rb') as f:
                with open(fname, 'r') as f:
                    predictivity = json.load(f)
                predictivity = np.array(predictivity).astype(int)
                union_concepts[layer_names[l]] += predictivity
        union_concepts[layer_names[l]] = union_concepts[layer_names[l]] > (timesteps * select_ratio)
        union_concepts[layer_names[l]] = union_concepts[layer_names[l]].astype('bool').astype('int')
        
    masks_save = os.path.join('eval_checkpoints_ap', args.concept, 'masks', str(wanda_thr[concept]))
    if not args.unskilled:
        print("------------------------- WARNING - Experiment wil run by removing unskilled neurons ------------------------------")
        masks_save = os.path.join('eval_checkpoints_unskilled', args.concept, 'masks')
    if not os.path.exists(masks_save):
        os.makedirs(masks_save)
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
            orig_weight = module.weight.data.clone().detach().cpu()
            # select second half of the weight
            in_feat, out_feat = orig_weight.shape
            weight_new = orig_weight[in_feat//2:, :].T
            # apply mask
            print(weight_new.shape, union_concepts[name].shape)
            # multiply each column of weight_new with union_concepts[name]
            weight_new *= (1 - union_concepts[name])
            weight_new = weight_new.T
            print("Layer: ", name, "Number of skilled neurons: ", np.mean(union_concepts[name]))
            weights = torch.zeros_like(orig_weight)
            weights[in_feat//2:, :] = weight_new
            weights[:in_feat//2, :] = orig_weight[:in_feat//2, :]
            weights = weights.to('cuda:6')
            module.weight.data = weights
            

            # save (masks[name]) to a file
            # with open(os.path.join(masks_save, f'{name}.pkl'), 'wb') as f:
            #     pickle.dump(union_concepts[name], f)
    
    # save checkpoint
    if not os.path.exists(f'eval_checkpoints_ap'):
        os.makedirs(f'eval_checkpoints_ap')
    torch.save(model.unet.state_dict(), f'eval_checkpoints_ap/{concept}.pt')


    # generate test image
    torch.manual_seed(gen_seed)
    np.random.seed(gen_seed)
    new_image = model(test_prompts[concept], safety_checker=safety_checker_).images[0]
    new_image.save(f'test_images/{concept}_{gen_seed}_new.png')



if __name__ == '__main__':
    main()