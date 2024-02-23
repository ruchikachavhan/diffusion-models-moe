import json
import os
import sys
import torch
import tqdm
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
import moe_utils
sys.path.append(os.getcwd())
import utils as dm_utils
from diffusers.models.activations import GEGLU


def get_model_block_config(model_id):
    config = {}
    # TODO - Make this automatic by reading keys and modules
    if model_id == 'runwayml/stable-diffusion-v1-5':
        config['down_blocks'] = {}
        config['down_blocks']['layer_idx'] = [0, 1, 2]
        config['down_blocks']['attention_idx'] = [0, 1]
        config['mid_block'] = {}
        config['mid_block']['layer_idx'] = [-1]
        config['mid_block']['attention_idx'] = [0]
        config['up_blocks'] = {}
        config['up_blocks']['layer_idx'] = [1, 2, 3]
        config['up_blocks']['attention_idx'] = [0, 1, 2]
    return config

    
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
    parser.add_argument('--num-neurons-expert', type=int, default=20, help='number of neurons in each expert')
    parser.add_argument('--templates', type=str, 
                        default='{}.{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight',
                        help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

    args = parser.parse_args()
    return args

def make_templates(args, config):
    template = args.templates.split(',')
    templates = []

    for t in template:
        for key in config.keys():
            for layer in config[key]['layer_idx']:
                if layer == -1:
                    t_ = '{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight'
                    for att in config[key]['attention_idx']:
                        templates.append(t_.format(key, att))
                else:
                    for att in config[key]['attention_idx']:
                        templates.append(t.format(key, layer, att))
    return templates

def test_template(templates, model):
    model_ffns = []
    for name, param in model.unet.named_modules():
        if 'ff.net' in name and isinstance(param, GEGLU):
            print(f"Found FFN: {name}")
            # append W1 of the FFN
            model_ffns.append(name + '.proj.weight')
    
    print(model_ffns, templates)
    #  assert that every element in the lists should be the same
    assert all([ffn in templates for ffn in model_ffns])
    print("All FFNs are considered for MOEfication. Test passed.")
            
def main():
    args = get_args()
    dm_utils.make_dirs(args)
    model, num_geglu = dm_utils.get_sd_model(args)
    model = model.to(args.gpu)
    block_config = get_model_block_config(args.model_id)

    torch.save(model.unet.state_dict(), os.path.join(args.res_path, args.model_id, 'moefication', 'model.pt'))

    config = moe_utils.ModelConfig(os.path.join(args.res_path, args.model_id, 'moefication', 'model.pt'), args.res_path, split_size=args.num_neurons_expert)

    templates = args.templates.split(',')
    templates = make_templates(args, block_config)
    test_template(templates, model)

    for template in templates:
        print(f"Splitting parameters for {template}")
        # For every FFN, we noe split the weights into clusters by using KMeansConstrained
        split = moe_utils.ParamSplit(config, template)
        split.split()
        split.cnt()
        split.save()


if __name__ == "__main__":
    main()



