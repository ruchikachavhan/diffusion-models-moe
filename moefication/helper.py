import os
import torch
import numpy as np
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

def make_templates(template, config):
    templates = []

    for key in config.keys():
        for layer in config[key]['layer_idx']:
            if layer == -1:
                t_ = '{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight'
                for att in config[key]['attention_idx']:
                    templates.append(t_.format(key, att))
            else:
                for att in config[key]['attention_idx']:
                    templates.append(template % (key, layer, att))
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

def modify_ffn(ffn, path, k):
    assert type(ffn) == GEGLU
    labels = torch.load(path)
    cluster_num = max(labels)+1
    patterns = []
    for i in range(cluster_num):
        patterns.append(np.array(labels) == i)
    # Shape of module.patterns is (number of total experts, number of neurons)
    # module.patterns[i, j] = 1 if neuron j is in expert i
    device = ffn.proj.weight.device
    dtype = ffn.proj.weight.dtype
    ffn.patterns = torch.Tensor(patterns).to(device).to(dtype)
    # ffn.k is the ratio of selected experts
    ffn.k = int(cluster_num * k)
    print("Moefied model with ", ffn.k, "experts in layer", path)


def modify_ffn_to_experts(model, args):
    # Modify FFN to add expert labels
    for name, module in model.unet.named_modules():
        if 'ff.net' in name and isinstance(module, GEGLU):
            ffn_name = name + '.proj.weight'
            path = os.path.join(args.res_path, 'param_split', ffn_name)
            modify_ffn(module, path, args.moefication['topk_experts'])
    
    return model

def initialise_expert_counter(model, timesteps=51):
    # Initialise expert frequency selection counter
    expert_counter = {}
    # make a expert counter for every time step
    for i in range(timesteps):
        expert_counter[i] = {}

    ffn_names_list = []
    # Modify FFN to add expert labels
    for name, module in model.unet.named_modules():
        if 'ff.net' in name and isinstance(module, GEGLU):
            ffn_name = name + '.proj.weight'
            for t in range(timesteps):
                expert_counter[t][ffn_name] = np.array([0.0 for _ in range(module.patterns.shape[0])])
            ffn_names_list.append(ffn_name)
    ffn_names_list.sort()

    return expert_counter, ffn_names_list