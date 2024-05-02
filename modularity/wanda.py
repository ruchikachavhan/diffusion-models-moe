import json
import os
import torch
import sys
import tqdm
from mod_utils import get_prompts
sys.path.append('moefication')
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
from neuron_receivers import Wanda
from PIL import Image 
import scipy
import pickle

def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    args.configure('modularity')

    # Model
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    print("Replce fn: ", replace_fn)
    model = model.to(args.gpu)

    # get the norm of the weights
    gate_weights = {}
    layer_names = []
    for name, module in model.text_encoder.named_modules():
            if isinstance(module, args.replace_fn) and 'mlp' in name and 'encoder.layers' in name:
                layer_names.append(name)
                weight = module.fc2.weight.detach().clone()
                gate_weights[name] = weight.abs().cpu()
                print("Name: ", name, weight.shape)
                
    # Make two separate norm calculator classes for base and adj prompts
    neuron_pred_base = Wanda(args.seed, args.timesteps, num_geglu, replace_fn = args.replace_fn, keep_nsfw = args.modularity['keep_nsfw'])
    neuron_pred_adj = Wanda(args.seed, args.timesteps, num_geglu, replace_fn = args.replace_fn, keep_nsfw = args.modularity['keep_nsfw'])

    base_prompts, adj_prompts, _, _ = get_prompts(args)
    norm_save_path = os.path.join(args.modularity['img_save_path'].split('images')[0])
     
    if not os.path.exists(os.path.join(norm_save_path, 'base_norms.pt')):
        # Saving norm values
        iter = 0
        for ann, ann_adj in tqdm.tqdm(zip(base_prompts, adj_prompts)):
            if iter >= 3 and args.dbg:
                break
            print("text: ", ann, ann_adj)
            
            neuron_pred_base.reset_layer()
            out, _ = neuron_pred_base.observe_activation(model, ann)

            neuron_pred_adj.reset_layer()
            out_adj, _ = neuron_pred_adj.observe_activation(model, ann_adj)
            # save images
            out.save(os.path.join(args.modularity['img_save_path'], f'base_{iter}.jpg'))
            out_adj.save(os.path.join(args.modularity['img_save_path'], f'adj_{iter}.jpg'))
            iter += 1
        
        act_norms_base, act_norms_adj = {}, {}
        for l in range(args.n_layers):
            act_norms_base[l] = neuron_pred_base.predictivity[l].get_column_norms()
            act_norms_adj[l] = neuron_pred_adj.predictivity[l].get_column_norms()
       
        torch.save(act_norms_base, os.path.join(norm_save_path, 'base_norms.pt'))
        torch.save(act_norms_adj, os.path.join(norm_save_path, 'adj_norms.pt'))
    else:
        act_norms_base = torch.load(os.path.join(norm_save_path, 'base_norms.pt'))
        act_norms_adj = torch.load(os.path.join(norm_save_path, 'adj_norms.pt'))
    sparsity_ratio = args.modularity['condition']['skill_ratio']
    print("Layer names: ", layer_names)

    for l in range(num_geglu):
        # wanda score is W.abs() * A
        metric_base = gate_weights[layer_names[l]] * act_norms_base[l]
        metric_adj = gate_weights[layer_names[l]] * act_norms_adj[l]

        # check inf
        if torch.isinf(act_norms_base[l]).any():
            print("Inf values in metric base")
            # metric_base[torch.isinf(metric_base)] = 0

        # do row-wise sorting for base in descending order
        _, sorted_idx = torch.sort(metric_base, dim=1, descending=True)
        pruned_indx = sorted_idx[:, :int(sparsity_ratio * metric_base.shape[1])].numpy()
            
        # do row-wise sorting for adj
        binary_mask_adj = torch.zeros_like(gate_weights[layer_names[l]])
        _, sorted_idx_adj = torch.sort(metric_adj, dim=1, descending=True)
        pruned_indx_adj = sorted_idx_adj[:, :int(sparsity_ratio * metric_adj.shape[1])].numpy()
        binary_mask_adj.scatter_(1, torch.tensor(pruned_indx_adj), 1)


        # # # make a binary mask of the size of weights 
        binary_mask = torch.zeros_like(gate_weights[layer_names[l]])
        diff = metric_adj > metric_base
        binary_mask = diff * binary_mask_adj
        binary_mask = binary_mask.float()

        print("Binary mask: ", binary_mask_adj.mean(), binary_mask_adj.shape)

        # convert binary mask to array
        binary_mask = binary_mask.cpu().numpy().astype(int)
        binary_mask = scipy.sparse.csr_matrix(binary_mask)
        print(binary_mask)

        # save in pickle file
        with open(os.path.join(args.modularity['skill_neuron_path'], f'layer_{l}.pkl'), 'wb') as f:
            pickle.dump(binary_mask, f)
        
        # print("Saving skilled weights in", os.path.join(args.modularity['skill_neuron_path'], f'layer_{l}.pt'))
        # torch.save(binary_mask_adj, os.path.join(args.modularity['skill_neuron_path'], f'layer_{l}.pt'))
            


if __name__ == '__main__':
    main()