import torch
import os
import json
import numpy as np
from diffusers.models.activations import GEGLU, GELU
from neuron_receivers.remove_wanda_neurons_fast import WandaRemoveNeuronsFast
import pickle
from PIL import Image

class MultiConceptRemoverWanda:
    def __init__(self, root, seed, T, n_layers, replace_fn = GEGLU, keep_nsfw=False, remove_timesteps=None, weights_shape=None, concepts_to_remove=None, wanda_thr=0.05):
        self.concepts_to_remove = concepts_to_remove
        # for every concept to remove initialise a WandaRemoveNeuronsFast object
        self.removers = {}
        self.seed = seed
        self.T = T
        self.n_layers = n_layers
        for concept in concepts_to_remove:
            print(f'Initialising remover for {concept} with threshold {wanda_thr[concept]}')
            path_expert_indx = os.path.join((root % (seed, concept)), f'skilled_neuron_wanda/{wanda_thr[concept]}')
            remover = WandaRemoveNeuronsFast(seed=seed, path_expert_indx = path_expert_indx, 
                    T=T, n_layers=n_layers, replace_fn=replace_fn, keep_nsfw=keep_nsfw,
                    remove_timesteps = remove_timesteps, weights_shape = weights_shape)
            self.removers[concept] = remover
        
        # initialise a Wanda neuron remover object 
        # Expert indices of this remover keep getting updated
        self.union_neuron_remover = WandaRemoveNeuronsFast(seed=seed, path_expert_indx = path_expert_indx,
                T=T, n_layers=n_layers, replace_fn=replace_fn, keep_nsfw=keep_nsfw,
                remove_timesteps = remove_timesteps, weights_shape = weights_shape)
    
    def reset_union_remover(self):
        print('Resetting union remover')
        self.union_neuron_remover.reset_time_layer()
        # empty the indices
        for i in range(0, self.union_neuron_remover.T):
            for j in range(0, self.union_neuron_remover.n_layers):
                self.union_neuron_remover.expert_indices[i][j] = np.zeros(self.union_neuron_remover.expert_indices[i][j].shape)

    def handle_multiple_concepts(self, concepts):
        # If we have two concepts to remove, we need to remove the neurons of both concepts
        # To do this, we take a union of self.expert_indices for both concepts
        # This way, we can remove the neurons of both concepts
        print(f'Handling multiple concepts: {concepts}')
        for c in concepts:
            self.removers[c].reset_time_layer()
            for i in range(0, self.removers[c].T):
                for j in range(0, self.removers[c].n_layers):
                    self.union_neuron_remover.expert_indices[i][j] = np.logical_or(self.union_neuron_remover.expert_indices[i][j], self.removers[c].expert_indices[i][j])
                    self.union_neuron_remover.expert_indices[i][j] = self.union_neuron_remover.expert_indices[i][j].astype(int)

    def remove_concepts(self, model, prompt, concepts):

        if len(concepts) == 0:
            # only run the model
            out_no_concept = model(prompt).images[0]
            return out_no_concept, None
        
        if len(concepts) > 1:
            self.handle_multiple_concepts(concepts)
            # remove the neurons of both concepts from the model
            self.union_neuron_remover.reset_time_layer()
            out_removal, _ = self.union_neuron_remover.observe_activation(model, prompt)
            single_removal_outputs = []
            for c in concepts:
                self.removers[c].reset_time_layer()
                im_, _ = self.removers[c].observe_activation(model, prompt)
                single_removal_outputs.append(im_)
        else:
            # remove the neurons of the concept from the model
            self.removers[concepts[0]].reset_time_layer()
            out_removal, _ = self.removers[concepts[0]].observe_activation(model, prompt)
            single_removal_outputs = []
        
        # generate original image
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        out_pre = model(prompt).images[0]

        # Stitch the images and return output
        out_pre = out_pre.resize((256, 256))
        out_removal = out_removal.resize((256, 256))
        output_image = Image.new('RGB', (530, 290))
        output_image.paste(out_pre, (0,40))
        output_image.paste(out_removal, (275,40))

        if len(single_removal_outputs) > 0:
            single_removal_im = []
            for i, im in enumerate(single_removal_outputs):
                new_im = Image.new('RGB', (530, 290))
                im = im.resize((256, 256))
                new_im.paste(out_pre, (0,40))
                new_im.paste(im, (275,40))
                single_removal_im.append(new_im)

        return output_image, single_removal_im if len(single_removal_outputs) > 0 else None




