import torch
import os
import json
import numpy as np
import scipy
import pickle


wanda_thr = {
    'Van Gogh': 0.02,
    'naked': 0.01,
    'english springer': 0.02,
    "chain saw": 0.05,
    'church': 0.05,
    'tench': 0.05,
    "golf ball": 0.05,
    'parachute': 0.05,
    'french horn': 0.05,
    'gas pump': 0.01,
    'parachute': 0.05,
    "cassette player": 0.05,
    'gas pump': 0.05
}

wanda_thr = {
    'Van Gogh': 0.02,
    'naked': 0.01,
    'english springer': 0.02,
    "chain saw": 0.05,
    'church': 0.05,
    'tench': 0.05,
    "golf ball": 0.05,
    'parachute': 0.05,
    'french horn': 0.05,
    'gas pump': 0.01,
    'parachute': 0.05,
    "cassette player": 0.05,
    'gas pump': 0.05
}
weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]

def main():

    timesteps = 51
    n_layers = 16
    seed = 0
    global_concepts = ['church', 'chain saw', 'tench', 'gas pump', 'french horn', 'parachute', 'english springer', 'golf ball', 'cassette player', 'gas pump'] 

    concepts = {}

    for concept in global_concepts:
        if concept == 'naked':
            concepts['naked'] = 'naked'
        if concept == 'Van Gogh':
            concepts['Van Gogh'] = 'Van Gogh'
        if concept in ['5artists', '10artists']:
            # read the file with 10 artist names
            with open(f'modularity/datasets/{concept}.txt', 'r') as f:
                artists = f.readlines()
                artists = [artist.strip() for artist in artists]
            
            for artist in artists:
                concepts[artist] = '5artists'
        if concept in ['Van Gogh']:
            concepts[concept] = 'Van Gogh'
        if concept in ['church', 'chain saw', 'tench', 'Gas Pump', 'french horn', 'parachute', 'english springer', 'golf ball', 'cassette player', 'gas pump']:
            concepts[concept] = concept
    
    print(concepts)

    union_concepts = {}
    for t in range(timesteps):
        union_concepts[t] = {}
        for l in range(n_layers):
            zeros = np.zeros(weights_shape[l])
            # convert to scipy sparse matrix
            union_concepts[t][l] = scipy.sparse.csr_matrix(zeros)


    select_ratio = 0.95
    print("Select ratio: ", select_ratio)

    root = 'results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/art/%s' 
    for c in concepts.keys():
        print(c)
        root = 'results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/art/%s' 
        if c == 'naked':
             root = 'results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s'
        
        path = os.path.join(root % (seed, c), 'skilled_neuron_wanda', str(wanda_thr[concepts[c]]))
        print("Reading from", path)

        for t in range(timesteps):
            for l in range(n_layers):
                with open(os.path.join(path, f'timestep_{t}_layer_{l}.pkl'), 'rb') as f:
                    # load sparse matrix
                    indices = pickle.load(f)
                    # take union
                    # out of the sparse matrix, only select 50% elements that are 1
                    indices = indices.toarray()
                    print("Sparisty before", np.mean(indices))
                    non_zero = np.where(indices != 0)
                    # if c == 'naked':
                    #     n = int(1 * len(non_zero[0]))
                    # else:
                    n = int(select_ratio * len(non_zero[0]))
                    random_choice = np.random.choice(len(non_zero[0]), n)
                    indices[non_zero[0][random_choice], non_zero[1][random_choice]] = 0
                    union_concepts[t][l] += indices
                    print("sparsity so far", np.mean(union_concepts[t][l]), np.mean(indices))

    output_path = root % (seed, "_".join(global_concepts))
    output_path = os.path.join(output_path, 'skilled_neuron_wanda', str(select_ratio))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # save union indices
    for t in range(timesteps):
        for l in range(n_layers):
            # if indices is still non-zero, consider it skilled
            indices = union_concepts[t][l].astype('bool').astype('int')
            print("Time step: ", t, "Layer: ", l, "Number of skilled neurons: ", np.mean(indices))
            with open(os.path.join(output_path, f'timestep_{t}_layer_{l}.pkl'), 'wb') as f:
                pickle.dump(indices, f)
            
        


if __name__ == '__main__':
    main()