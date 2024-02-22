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
from utils import make_dirs, get_sd_model
from relufy_model import NeuronReceiver

def coco_dataset(data_path, split, num_images=1000):
    with open(os.path.join(data_path, f'annotations/captions_{split}2014.json')) as f:
        data = json.load(f)
    data = data['annotations']
    # select 30k images randomly
    np.random.seed(0)
    np.random.shuffle(data)
    data = data[:num_images]
    imgs = [os.path.join(data_path, f'{split}2014', 'COCO_' + split + '2014_' + str(ann['image_id']).zfill(12) + '.jpg') for ann in data]
    anns = [ann['caption'] for ann in data]
    return imgs, anns

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
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    make_dirs(args)
    
    model, num_geglu = get_sd_model(args)
    model = model.to(args.gpu)

    imgs, anns = coco_dataset(args.data_path, 'val', args.num_images)

    # Neuron receiver to store gates for every sample
    neuron_receiver = NeuronReceiver()
    if args.fine_tuned_unet is not None:
        neuron_receiver.test(model)
        print("Neuron receiver test passed")

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 10 and args.dbg:
            break

        print("Iter: ", iter)
        print("text: ", ann)
        out, gates = neuron_receiver.observe_activation(model, img, ann)
        if iter < 10:
            # Save some images
            out.save(os.path.join(args.res_path, 'images', args.model_id, f'img_{iter}.jpg'))
       
        all_results = {}
        all_results['time_steps'] = {}
        for t in range(args.timesteps):
            all_results['time_steps'][t] = {}
            all_results['time_steps'][t]['exact_zero_ratio'] = [0.0 for _ in range(num_geglu)]

        # divide gate into chunks of number of time steps
        for i in range(0, len(gates), num_geglu):
            gate_timestep = gates[i:i+num_geglu]
            for j, gate in enumerate(gate_timestep):
                if j > num_geglu:
                    continue
                # check sparsity
                # check if values of the gate == 0
                # if yes, then it is a sparse neuron
                # calculate ratio of sparse neurons per layer for each time step

                mask = gate == 0.0
                # % of neurons that are 0 out of total neurons (= hidden dimension)
                exact_zero_ratio = mask.int().sum(-1).float() / gate.shape[-1]
                # Take mean over all tokens
                exact_zero_ratio = exact_zero_ratio.mean()
                all_results['time_steps'][i//num_geglu]['exact_zero_ratio'][j] = exact_zero_ratio.item()
                    
        with open(os.path.join(args.res_path, 'sparsity', args.model_id, f'sparsity_{iter}.json'), 'w') as f:
            json.dump(all_results, f)

        iter += 1
        
    
if __name__ == '__main__':
    main()

