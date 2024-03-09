import json
import torch
import os
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import yaml
from collections import Counter

def configure_exp(config):
    # set result paths
    seeds = config['seeds']['range']
    results_folders = []
    if config['fine_tuned_unet'] is not None:
        config['res_path'] = os.path.join(config['res_path'], 'fine-tuned-relu')
    for seed in seeds:
        if seed == 0:
            root_path = os.path.join(config['res_path'], config['model_id'])
        else:
            new_path = f'results_seed_{seed}' + '/' + config['res_path'].split('/')[1]
            root_path = os.path.join(new_path, 'fine-tuned-relu' if config['fine_tuned_unet'] is not None else 'baseline')
            root_path = os.path.join(root_path, config['model_id'])
        results_folders.append(os.path.join(root_path, 'modularity', config['modularity']['adjective'], 
                        f'skilled_neuron_{config["set_to_average"]["name"]}_{config["set_to_average"]["ratio"]}', 
                        'with_bounding_boxes' if config['modularity']['bounding_box'] else ''))
        
    # set save paths
    exp_name, ratio = config['set_to_average']['name'], config['set_to_average']['ratio']
    save_path = os.path.join('results_all_seeds/stable-diffusion',
                            'fine-tuned-relu' if config['fine_tuned_unet'] is not None else 'baseline',
                            config['model_id'], 'modularity', 
                            config['modularity']['adjective'], f'skilled_neuron_{exp_name}_{ratio}', 
                            'with_bounding_boxes' if config['modularity']['bounding_box'] else '')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return results_folders, save_path

def main():
    # read config file with yaml
    with open('experiments/intersection_over_seeds.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # configure experiment, make directories
    results_folders, save_path = configure_exp(config)
    print(results_folders, save_path)

    intersection_dict = {}
    for t in range(config['timesteps']):
        intersection_dict[t] = {}
        for l in range(config['n_layers']):
            intersection_dict[t][l] = set()
        
    # read files for different seeds
    iter = 0
    for folder in results_folders:
        for t in range(config['timesteps']):
            for l in range(config['n_layers']):
                # read json files
                with open(os.path.join(folder, f'timestep_{t}_layer_{l}.json')) as f:
                    data = json.load(f)
                    if iter == 0:
                        intersection_dict[t][l] = set(data)
                    else:
                        intersection_dict[t][l] = intersection_dict[t][l].intersection(set(data))
                    # print(f"timestep {t}, layer {l}: {intersection_dict[t][l]}")
                    # intersection_dict[t][l] += data
        iter += 1

    # save the experts for every time step and layer
    for t in range(config['timesteps']):
        for l in range(config['n_layers']):
            with open(os.path.join(save_path, f'timestep_{t}_layer_{l}.json'), 'w') as f:
                json.dump(list(intersection_dict[t][l]), f)
    
    # for every time step and layer, count the frequency of each expert occuring in all seeds
    # for t in range(config['timesteps']):
    #     for l in range(config['n_layers']):
    #         intersection_dict[t][l] = Counter(intersection_dict[t][l])
    #         # select the experts with top 50% frequency
    #         # sort the counter keys according to values
    #         sorted_experts = sorted(intersection_dict[t][l], key=intersection_dict[t][l].get, reverse=True)
    #         # select the top 40% experts
    #         topk_experts = sorted_experts[:int(len(sorted_experts)* 0.8)]
    #         intersection_dict[t][l] = topk_experts
    #         print(f"timestep {t}, layer {l}: {intersection_dict[t][l]}")
    
    #         # save the experts
    #         with open(os.path.join(save_path, f'timestep_{t}_layer_{l}.json'), 'w') as f:
    #             json.dump(intersection_dict[t][l], f)



if __name__ == "__main__":
    main()