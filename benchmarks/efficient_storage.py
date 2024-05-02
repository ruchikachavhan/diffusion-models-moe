import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
sys.path.append(os.getcwd())
import utils
from diffusers.models.activations import LoRACompatibleLinear, GEGLU
import pandas as pd
import scipy
import pickle

weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]

concepts = {
    # 'Alex Alemany,painter': 'artist',
    # 'John Howe,illustrator': 'artist',
    # 'Antonio J. Manzanedo': 'artist',
    # 'Alpo Jaakola,painter': 'artist',
    # 'Abraham Mintchine,painter': 'artist',
    # 'Apollinary Vasnetsov,landscape art,history painting': 'artist',
    # 'John Constable,realism,landscape art': 'artist',
    # 'Johannes Vermeer,genre painting,portrait': 'artist',
    # 'Nicolas Mignard,mythological painting,painter': 'artist',
    # 'John Whitcomb': 'artist',
    # 'Amedeo Modigliani,landscape art,portrait': 'artist',
    # 'Jordan Grimmer': 'artist',
    # 'A.J.Casson': 'artist',
    # 'Akira Toriyama,mangaka,fantasy': 'artist',
    # 'Salvador Dali': 'artist',
    # 'Greg Rutkowski': 'artist',
    # 'Jeremy Mann': 'artist',
    # 'Van Gogh': 'artist',
    # 'Monet': 'artist',
    # 'Pablo Picasso': 'artist',
    # 'naked': 'naked'
    # 'memorize': 'memorize'
    'gender_female': 'gender_female'
}


def main():
    concepts_list = concepts.keys()
    timesteps = 51
    n_layers = 16
    for c in concepts_list:
        root = f'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/{c}/skilled_neuron_wanda/0.003'
        # read indices in the file
        print(root)

        files = os.listdir(root)
        files = [f for f in files if f.endswith('.json')]
        # for every weight shape, we need to create a dataframe 
        weights_df = {}
        for i in range(n_layers):
            weights_df[i] = {}
        for t in range(timesteps):
            for l in range(n_layers):
                path = os.path.join(root, f'timestep_{t}_layer_{l}.json')
                with open(path, 'r') as f:
                    indices = json.load(f)
                    indices = torch.tensor(indices)
                binary_mask = torch.zeros(weights_shape[l])
                binary_mask[indices[:, 0], indices[:, 1]] = 1
                # store binary mask in the datframe such that every column starts with timestep and layer
                binary_mask = binary_mask.numpy().astype(int)
                print(f'timestep_{t}_layer_{l}', binary_mask.shape)
                weights_df[l][f'timestep_{t}_layer_{l}'] = binary_mask    
        
        # save the dataframe for all the weights
        for i in range(n_layers):
            for keys in weights_df[i].keys():
                values = weights_df[i][keys]
                values = scipy.sparse.csr_matrix(values)
                print(values)
                # save in pickle file
                print(os.path.join(root, f'{keys}.pkl'))
                with open(os.path.join(root, f'{keys}.pkl'), 'wb') as f:
                    pickle.dump(values, f)

if __name__ == '__main__':
    main()