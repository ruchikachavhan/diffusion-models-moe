import json
import os
import torch
import sys
import tqdm
import numpy as np
from mod_utils import get_prompts
sys.path.append('moefication')
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
from neuron_receivers import Wanda
from PIL import Image 
from diffusers.models.activations import LoRACompatibleLinear
from transformers.models.clip.modeling_clip import CLIPMLP
import scipy
import pickle

def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    # args.configure('modularity')

    # Input topk experts for evaluation
    concept = str(sys.argv[1])
    if concept is not None:
        args.modularity['adjective'] = str(concept).strip()
    print(f"Adjective: {args.modularity['adjective']}")

    # check if a second argument exists
    if len(sys.argv) > 2:
        args.modularity['file_name'] = str(sys.argv[2])
        args.modularity['adjective'] = str(sys.argv[2])

    
    print(f"Adjective: {args.modularity['adjective']}")
    print(f"File name: {args.modularity['file_name']}")

    args.configure('modularity')

    # Model
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    print("Replce fn: ", replace_fn)
    model = model.to(args.gpu)

    # get the norm of the weights
    gate_weights = {}
    layer_names = []

    if args.hook_module == 'unet':
        for name, module in model.unet.named_modules():
            if isinstance(module, LoRACompatibleLinear) and 'ff.net' in name and not 'proj' in name:
            # if isinstance(module, args.replace_fn) and 'ff.net' in name:
                layer_names.append(name)
                print("Name: ", name, module.weight.shape)
                weight = module.weight.detach()
                gate_weights[name] = weight.abs().cpu()

    elif args.hook_module == 'text':
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, CLIPMLP) and 'mlp' in name and 'encoder.layers' in name:
                layer_names.append(name)
                weight = module.fc2.weight.detach().clone()
                gate_weights[name] = weight.abs().cpu()
                print("Name: ", name, weight.shape)
                
    # Make two separate norm calculator classes for base and adj prompts
    neuron_pred_base = Wanda(args.seed, args.timesteps, num_geglu, replace_fn = args.replace_fn, keep_nsfw = args.modularity['keep_nsfw'], hook_module=args.hook_module)
    neuron_pred_adj = Wanda(args.seed, args.timesteps, num_geglu, replace_fn = args.replace_fn, keep_nsfw = args.modularity['keep_nsfw'], hook_module=args.hook_module)

    base_prompts, adj_prompts, _ = get_prompts(args)
    norm_save_path = os.path.join(args.modularity['img_save_path'].split('images')[0])
    print("Norms of the activations in: ", norm_save_path)
    if not os.path.exists(os.path.join(norm_save_path, 'base_norms.pt')):
    # if True:
        # Saving norm values
        iter = 0
        for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
            if iter >= 3 and args.dbg:
                break
            print("text: ", ann, ann_adj)

            # select random seed
            if args.modularity['adjective'] in ['gender', 'nudity', 'violence', 'gender_female','Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'garbage truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']:
                seed = torch.randint(0, 250, (1,)).item()
                neuron_pred_base.seed = seed
                neuron_pred_adj.seed = seed
                print("Seed: ", seed)
            else:
                print("Seed: ", args.seed)
            
            neuron_pred_base.reset_time_layer()
            out, _ = neuron_pred_base.observe_activation(model, ann)

            neuron_pred_adj.reset_time_layer()
            out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj)
            # save images
            if iter < 5:
                print("Saving images", os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
                out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
                out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
            
            iter += 1
        
        # get the norms
        if args.hook_module == 'unet':
            act_norms_base = neuron_pred_base.predictivity.get_column_norms()
            act_norms_adj = neuron_pred_adj.predictivity.get_column_norms()
            # save
            neuron_pred_base.predictivity.save(os.path.join(norm_save_path, 'base_norms.pt'))
            neuron_pred_adj.predictivity.save(os.path.join(norm_save_path, 'adj_norms.pt'))
        elif args.hook_module == 'text':
            args.timesteps = 1
            act_norms_base, act_norms_adj = {}, {}
            for t in range(args.timesteps):
                act_norms_base[t] = {}
                act_norms_adj[t] = {}
                for l in range(num_geglu):
                    act_norms_base[t][l] = neuron_pred_base.predictivity[l].get_column_norms()
                    act_norms_adj[t][l] = neuron_pred_adj.predictivity[l].get_column_norms()

            # save
            torch.save(act_norms_base, os.path.join(norm_save_path, 'base_norms.pt'))
            torch.save(act_norms_adj, os.path.join(norm_save_path, 'adj_norms.pt'))
        
    else:
        act_norms_base = torch.load(os.path.join(norm_save_path, 'base_norms.pt'))
        act_norms_adj = torch.load(os.path.join(norm_save_path, 'adj_norms.pt'))
        print("Loaded norms from: ", os.path.join(norm_save_path, 'base_norms.pt'))

    # sort layer names - very important to get the right layer names
    layer_names.sort()
    sparsity_ratio = args.modularity['condition']['skill_ratio']
    print("Layer names: ", layer_names)

    timesteps = 1 if args.hook_module == 'text' else args.timesteps

    print(act_norms_adj.keys(), num_geglu, timesteps)

    for t in range(timesteps):
        for l in range(num_geglu):
            # wanda score is W.abs() * A
            metric_base = gate_weights[layer_names[l]] * act_norms_base[t][l]
            metric_adj = gate_weights[layer_names[l]] * act_norms_adj[t][l]

            # check for inf values here
            if torch.isinf(metric_base).any():
                print("Inf values in metric base")

            # do row-wise sorting for base in descending order
            _, sorted_idx = torch.sort(metric_base, dim=1, descending=True)
            pruned_indx = sorted_idx[:, :int(sparsity_ratio * metric_base.shape[1])].numpy()
            
            # # do row-wise sorting for adj
            binary_mask_adj = torch.zeros_like(gate_weights[layer_names[l]])
            _, sorted_idx_adj = torch.sort(metric_adj, dim=1, descending=True)
            pruned_indx_adj = sorted_idx_adj[:, :int(sparsity_ratio * metric_adj.shape[1])].numpy()
            binary_mask_adj.scatter_(1, torch.tensor(pruned_indx_adj), 1)


            # # make a binary mask of the size of weights 
            binary_mask = torch.zeros_like(gate_weights[layer_names[l]])
            diff = metric_adj > metric_base
            binary_mask = diff * binary_mask_adj
            binary_mask = binary_mask.float()

            # convert binary mask to array
            binary_mask = binary_mask.cpu().numpy().astype(int)
            binary_mask = scipy.sparse.csr_matrix(binary_mask)
            print("Binary mask shape: ", binary_mask.shape, np.mean(binary_mask.toarray()))
            # save in pickle file
            with open(os.path.join(args.modularity['skill_neuron_path'], f'timestep_{t}_layer_{l}.pkl'), 'wb') as f:
                pickle.dump(binary_mask, f)

            # save unskilled neurons as well
            # binary_mask = torch.zeros_like(gate_weights[layer_names[l]])
            # diff = metric_base > metric_adj
            # binary_mask = diff * binary_mask_adj
            # binary_mask = binary_mask.float()
            # binary_mask = binary_mask.cpu().numpy().astype(int)
            # binary_mask = scipy.sparse.csr_matrix(binary_mask)
            # print("unskilled Binary mask shape: ", binary_mask.shape, np.mean(binary_mask.toarray()))

            # # save in pickle file
            # unskilled_path = args.modularity['skill_neuron_path'].replace('skilled', 'unskilled')
            # print("Unskilled path: ", unskilled_path)
            # if not os.path.exists(unskilled_path):
            #     os.makedirs(unskilled_path)

            # # save
            # with open(os.path.join(unskilled_path, f'timestep_{t}_layer_{l}.pkl'), 'wb') as f:
            #     pickle.dump(binary_mask, f)


            


if __name__ == '__main__':
    main()