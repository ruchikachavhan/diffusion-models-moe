import os
import sys
import yaml
import numpy as np
import torch
import json
import tqdm
from PIL import Image
from torchvision import transforms
sys.path.append(os.getcwd())
from utils import get_sd_model, coco_dataset
from eval_coco import calculate_fid
from neuron_receivers import FrequencyMeasure
sys.path.append('moefication')
import moe_utils
from helper import modify_ffn_to_experts
sys.path.append('comparing_two_models')
from config_utils import Config
import matplotlib.pyplot as plt

def  make_expert_counter(timesteps, ffn_names_list, num_experts_per_ffn):
    # bad solution for averaging but had to do it
    expert_counter = {}
    for t in range(timesteps):
        expert_counter[t] = {}
        for ffn_name in ffn_names_list:
            expert_counter[t][ffn_name] = [0] * num_experts_per_ffn[ffn_name]
    
    return expert_counter
def main():

    args1 = Config('experiments/compare_two_models.yaml', 'compare_peft')  
    args2 = Config('experiments/compare_two_models.yaml', 'compare_peft')  
    args = [args1, args2]

    # load the two models
    # number of models to compare
    model_paths = args[0].fine_tuned_unet.split(',')
    model_names = args[0].model_names.split(',')
    print(f"Model paths: {model_paths}", f"Model names: {model_names}")
    models = []

    for i, model_path in enumerate(model_paths):
        args[i].fine_tuned_unet = model_path
        args[i].model_name = model_names[i]
        model, num_geglu = get_sd_model(args[i])
        model = model.to(args[i].gpu)
        models.append(model)
        if not os.path.exists(os.path.join(args[i].save_path, model_names[i], 'param_split')):
            os.makedirs(os.path.join(args[i].save_path, model_names[i], 'param_split'))

    # load the dataset
    imgs, anns = coco_dataset(args[0].dataset['path'], 'val', args[0].inference['num_images'])  

    # read convert ffns to experts
    for i, model in enumerate(models):
        args[i].res_path = os.path.join(args[i].res_path, model_names[i])
        model, ffn_names_list, num_experts_per_ffn = modify_ffn_to_experts(model, args[i])
        models[i] = model
    
    neuron_receiver = FrequencyMeasure(args[0].seed, args[0].timesteps, 
                                args[0].n_layers, num_experts_per_ffn, ffn_names_list)
    
    iter = 0

    if not os.path.exists(os.path.join(args[0].save_path, 'expert_counters.json')):
        expert_counter = [make_expert_counter(args[0].timesteps, ffn_names_list, num_experts_per_ffn)
                        for _ in range(len(models))]

        for img, ann in tqdm.tqdm(zip(imgs, anns)):
            if iter > 5 and args[0].dbg:
                break
            print("text: ", ann)
            for i, model in enumerate(models):
                neuron_receiver.reset()
                out_moe, _ = neuron_receiver.observe_activation(model, ann)
                label_counter = neuron_receiver.label_counter

                # add to expert counter
                for t in range(args[0].timesteps):
                    for ffn_name in ffn_names_list:
                        for expert in range(num_experts_per_ffn[ffn_name]):
                            expert_counter[i][t][ffn_name][expert] += \
                                label_counter[t][ffn_names_list.index(ffn_name)][expert] / args[0].inference['num_images']
            iter += 1
        # save expert   counter
        with open(os.path.join(args[i].save_path, 'expert_counters.json'), 'w') as f:
            json.dump(expert_counter, f)
    else:
        with open(os.path.join(args[i].save_path, 'expert_counters.json'), 'r') as f:
            expert_counter = json.load(f)
    print(expert_counter[0].keys())

    # Compare most commonly selected experts
    set_diff = {}
    for t in range(args[0].timesteps):
        set_diff[t] = {}
        for ffn_name in ffn_names_list:
            set_diff[t][ffn_name] = []

    t_range = [0, 10, 20, 30, 40, 50]
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    # put hspace between subplots
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('Set intersecton ratio in expert selection')
    for t in range(args[0].timesteps):
        ratios = []
        for ffn_name in ffn_names_list:
            list_freq1 = expert_counter[0][str(t)][ffn_name]
            list_freq2 = expert_counter[1][str(t)][ffn_name]
            # select top - 70% of the experts
            topk = int(0.4 * len(list_freq1))
            # do argmax in descending order for topk
            # argsort in descending order 
            list_freq1 = np.argsort(list_freq1)[::-1][:topk]
            list_freq2 = np.argsort(list_freq2)[::-1][:topk]

            # find the difference in the sets
            set_diff[t][ffn_name] = set(list_freq1).intersection(set(list_freq2))
            ratio_set_diff = len(set_diff[t][ffn_name]) / len(list_freq1)
            print(f"ratio of set difference - {ffn_name}: {ratio_set_diff}")
            ratios.append(ratio_set_diff)
            # plot the set difference for each timestep
        if t in t_range:
            axes = ax[(t//10)//2][(t//10)%2]
            axes.bar(range(len(ratios)), ratios)
            axes.set_title(f"t = {t}")
            axes.set_xticks(range(len(ratios)))
            axes.set_xlabel('Layer ID')
            axes.set_ylabel('Set intersection ratio')
            axes.set_ylim(0.8, 1.0)
            plt.savefig(os.path.join(args[i].save_path, f'set_difference.png'))

                
if __name__ == "__main__":
    main()