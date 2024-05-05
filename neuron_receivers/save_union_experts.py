import torch
import numpy as np
import os
import pickle
import json
import scipy

concepts_union = {
    'Van Gogh', 
    'naked'
}

thr_dict = {
    'Van Gogh': 0.02,
    'naked': 0.01
}

weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]

def main():
    # take union of all expert indices for cooncepts

    root_path = 'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/'
    time_steps = 51
    n_layers = 16

    # Initiliase the union indices with scipy sparse matrices
    concept_union_indices = {}
    for t in range(time_steps):
        concept_union_indices[t] = {}
        for l in range(n_layers):
            concept_union_indices[t][l] = np.zeros(weights_shape[l])
            zeros = np.zeros(weights_shape[l])
            concept_union_indices[t][l] = scipy.sparse.csr_matrix(zeros)


    for concept in concepts_union.keys():
        path = os.path.join(root_path, concept, 'skilled_neuron_wanda', str(thr_dict[concept]))

        # read pickle files
        for t in range(time_steps):
            for l in range(n_layers):
                file_path = os.path.join(path, f'timestep_{t}_layer_{l}.pkl')
                with open(file_path, 'rb') as f:
                    # load the sparse matrix
                    binary_mask = pickle.load(f)
                    # union of the binary masks
                    concept_union_indices[t][l] += binary_mask

    # save the union indices
    for t in range(time_steps):
        for l in range(n_layers):
            print("Sparsity", concept_union_indices[t][l].toarray().mean())
            # with open(os.path.join(root_path, 'union', f'timestep_{t}_layer_{l}.pkl'), 'wb') as f:
            #     pickle.dump(concept_union_indices[t][l], f)
               
        

if __name__ == '__main__':
    main()
