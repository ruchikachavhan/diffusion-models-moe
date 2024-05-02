import os
import sys
import yaml
import numpy as np
import torch
import json
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import PixArtAlphaPipeline
sys.path.append('sparsity')
from diffusers.models.activations import GEGLU, GELU
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from relufy_model import find_and_change_geglu
from transformers.models.clip.modeling_clip import CLIPMLP
from sld import SLDPipeline

def make_dirs(args):
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    # make directory for model id
    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'images')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'images'))
        os.makedirs(os.path.join(args.res_path, args.model_id, 'images', 'evaluation_coco'))

    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'sparsity')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'sparsity'))

    if args.moefication is not None:
        # make image directory
        if not os.path.exists(os.path.join(args.res_path, args.model_id)):
            os.makedirs(os.path.join(args.res_path, args.model_id))
            os.makedirs(os.path.join(args.res_path, args.model_id, 'moefication'))

    if args.modularity is not None:
        if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'])):
            os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective']))

        if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], 'images')):
            os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], 'images'))

        if not os.path.exists(args.modularity['img_save_path']):
            os.makedirs(args.modularity['img_save_path'])

        if args.modularity['concept_removal']:
            if not os.path.exists(args.modularity['skill_expert_path']):
                os.makedirs(args.modularity['skill_expert_path'])
            if not os.path.exists(args.modularity['remove_expert_path']):
                os.makedirs(args.modularity['remove_expert_path'])
            if not os.path.exists(args.modularity['remove_expert_path_val']):
                os.makedirs(args.modularity['remove_expert_path_val'])
            if args.modularity['skill_neuron_path'] is not None:
                if not os.path.exists(args.modularity['skill_neuron_path']):
                    os.makedirs(args.modularity['skill_neuron_path'])
                if not os.path.exists(args.modularity['remove_neuron_path']):
                    os.makedirs(args.modularity['remove_neuron_path'])
                if not os.path.exists(args.modularity['remove_neuron_path_val']):
                    os.makedirs(args.modularity['remove_neuron_path_val'])
            if not os.path.exists(args.modularity['plots']):
                os.makedirs(args.modularity['plots'])
        

def get_sd_model(args):

    if 'v1-5' in args.model_id:
        if args.fine_tuned_unet is not None:
            print("Loading from fine-tuned checkpoint at", args.fine_tuned_unet)
            # Upload pre-trained relufied model
            model_path = args.fine_tuned_unet
            unet = UNet2DConditionModel.from_pretrained(model_path + "unet", torch_dtype=torch.float16)
            # change geglu to relu
            unet = find_and_change_geglu(unet)
            model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16)
        else:
            print("Loading from pre-trained model", args.model_id)
            model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        num_geglu = args.n_layers

        replace_fn = GEGLU
    elif args.model_id == 'CompVis/stable-diffusion-v1-4':
        model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        num_geglu = args.n_layers
        replace_fn = GEGLU

    elif args.model_id == 'CompVis/stable-diffusion-v1-4-safe':
        model = SLDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        num_geglu = args.n_layers
        replace_fn = GEGLU

    elif args.model_id == "stabilityai/stable-diffusion-2":
        scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
        model = StableDiffusionPipeline.from_pretrained(args.model_id, scheduler=scheduler, torch_dtype=torch.float16)
        num_geglu = args.n_layers
        replace_fn = GEGLU
    
    elif args.model_id == "stabilityai/stable-diffusion-2-1":
        model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
        num_geglu = args.n_layers
        replace_fn = GEGLU
    elif args.model_id == 'UCE-baseline':
        unet_path = "../unified-concept-editing/models/erased-violence_nudity_harm-towards_uncond-preserve_false-sd_1_4-method_replace.pt"
        unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(unet_path))
        model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet, torch_dtype=torch.float16)
        num_geglu = args.n_layers
        replace_fn = GEGLU
                                    

    elif 'xl-base-1.0' in args.model_id:
        model = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float32)
    
    elif 'PixArt-alpha' in args.model_id:
        model = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16)
        model.enable_model_cpu_offload()
        num_geglu = 28
        # HACK, make a unet module in the model
        model.unet = model.transformer
        replace_fn = GELU
    
    elif 'lcm-sdxl' in args.model_id:
        unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
        model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
        model.scheduler = LCMScheduler.from_config(model.scheduler.config)
        num_geglu = 0
        for name, module in model.unet.named_modules():
            if 'ff.net' in name and isinstance(module, GEGLU):
                num_geglu += 1
        replace_fn = GEGLU
        print("Number of GEGLU layers", num_geglu)

    return model, num_geglu, replace_fn

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


class Config:
    def __init__(self, path, exp_name):
        # Load config file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        self.exp_name = exp_name
        for key, value in config.items():
            setattr(self, key, value)
        
        if self.seed!='all':
            self.res_path = f'results/results_seed_{self.seed}' + '/' + self.res_path.split('/')[1]
        elif self.seed == 'all':
            self.res_path = 'results/results_all_seeds' + '/' + self.res_path.split('/')[1]
            self.seed = self.default_eval_seed
    
        # change result directory
        if self.fine_tuned_unet is not None:
            self.res_path = os.path.join(self.res_path, 'fine-tuned-relu')
        else:
            self.res_path = os.path.join(self.res_path, 'baseline')
        print("Saving results to", self.res_path)
        self.save_path = os.path.join(self.res_path, self.model_id, exp_name)


    def configure(self, exp_name):     

        # Folders fo modularity experiments 
        if exp_name == 'modularity':  
            self.save_path = os.path.join(self.res_path, self.model_id, exp_name, self.modularity['adjective'])

        if self.modularity is not None:
            self.modularity['img_save_path'] = os.path.join(self.save_path, 'images')
            ratio = self.modularity['condition']['skill_ratio']
            condition = self.modularity['condition']['name'] 
            if self.modularity['single_sample_test']:
                prefix = 'single_sample_test'
            else:
                prefix = ''
            if self.modularity['concept_removal']:
                if self.modularity['condition']['name'] in ['t_test', 'wanda']:
                    self.modularity['skill_expert_path'] = os.path.join(self.save_path, prefix, f'skilled_expert_{condition}', str(ratio))
                    self.modularity['skill_neuron_path'] = os.path.join(self.save_path, prefix, f'skilled_neuron_{condition}', str(ratio))
                elif self.modularity['condition']['name'] == 'moefy_compare':
                    topk_experts = self.moefication['topk_experts']
                    self.modularity['skill_expert_path'] = os.path.join(self.save_path, prefix, f'skilled_expert_{condition}', str(topk_experts))
                    self.modularity['skill_neuron_path'] = None
                elif self.modularity['condition']['name'] in ['t_test_expert', 't_test_moefy_compare', 't_test_t_test_expert', 't_test_expert_moefy_compare', 't_test_intersection', 't_test_union', 't_test_expert_union']:
                    self.modularity['skill_expert_path'] = os.path.join(self.save_path, prefix, f'skilled_expert_{condition}', str(ratio))
                    self.modularity['skill_neuron_path'] = None

                if self.modularity['bounding_box']:
                    self.modularity['skill_expert_path'] = os.path.join(self.modularity['skill_expert_path'], 'with_bounding_boxes')

                self.modularity['remove_expert_path'] = os.path.join(self.modularity['skill_expert_path'], 'remove_experts')
                self.modularity['remove_expert_path_val'] = os.path.join(self.modularity['skill_expert_path'], 'remove_experts_val')
                if self.modularity['skill_neuron_path'] is not None:
                    self.modularity['remove_neuron_path'] = os.path.join(self.modularity['skill_neuron_path'], 'remove_neurons')
                    self.modularity['remove_neuron_path_val'] = os.path.join(self.modularity['skill_neuron_path'], 'remove_neurons_val')

                self.modularity['plots'] = os.path.join(self.modularity['skill_expert_path'], 'plots')

        
        # set experiment folders
        make_dirs(self)
        
    def __repr__(self):
        for key, value in self.config.items():
            print(f"{key}: {value}")
    
class Average:
    '''
    Class to measure average of a set of values
    for all timesteps and layers
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class StandardDev:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)

    def stddev(self):
        return self.variance() ** 0.5


class StatMeter:
    '''
    Class to measure average and standard deviation of a set of values
    for all timesteps and layers
    '''
    def __init__(self, T, n_layers):
        self.reset()
        self.results = {}
        self.results['time_steps'] = {}
        self.T = T
        self.n_layers = n_layers
        for t in range(T):
            self.results['time_steps'][t] = {}
            for i in range(n_layers):
                self.results['time_steps'][t][i] = {}
                self.results['time_steps'][t][i]['avg'] = Average()
                self.results['time_steps'][t][i]['std'] = StandardDev()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, t, n_layer):
        self.results['time_steps'][t][n_layer]['avg'].update(val)
        self.results['time_steps'][t][n_layer]['std'].update(val)
        
    
    def save(self, path):
        for t in range(self.T):
            for i in range(self.n_layers):
                self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].avg
                self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].stddev()
                # check if its and array
                if isinstance(self.results['time_steps'][t][i]['avg'], np.ndarray):
                    self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].tolist()
                if isinstance(self.results['time_steps'][t][i]['std'], np.ndarray):
                    self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].tolist()

        with open(path, 'w') as f:
            json.dump(self.results, f)

import numpy as np

class ColumnNormCalculator:
    def __init__(self):
        '''
        Calculated Column Norm of a matrix incrementally as rows are added
        Assumes 2D matrix
        '''
        self.A = np.zeros((0, 0))
        self.column_norms = torch.tensor([])

    def add_rows(self, rows):
        if len(self.A) == 0:  # If it's the first row
            self.A = rows
            self.column_norms = torch.norm(self.A, dim=0)
        else:
            # self.A = np.vstack((self.A, rows))
            new_row_norms = torch.norm(rows, dim=0)
            self.column_norms = torch.sqrt(self.column_norms**2 + new_row_norms**2)

    def get_column_norms(self):
        return self.column_norms



class TimeLayerColumnNorm:
    '''
    Column Norm calculator for all timesteps and layers
    '''
    def __init__(self, T, n_layers):
        self.T = T
        self.n_layers = n_layers
        self.column_norms = {}
        for t in range(T):
            self.column_norms[t] = {}
            for i in range(n_layers):
                self.column_norms[t][i] = ColumnNormCalculator()

    def update(self, rows, t, n_layer):
        self.column_norms[t][n_layer].add_rows(rows)

    def get_column_norms(self):
        results = {}
        for t in range(self.T):
            results[t] = {}
            for i in range(self.n_layers):
                results[t][i] = self.column_norms[t][i].get_column_norms()
        return results
    
    def save(self, path):
        results = self.get_column_norms()
        torch.save(results, path)