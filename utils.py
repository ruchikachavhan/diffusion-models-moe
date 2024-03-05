import os
import sys
import yaml
import numpy as np
import torch
import json
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
sys.path.append('sparsity')
from relufy_model import find_and_change_geglu

def make_dirs(args):
    if not os.path.exists('test_images'):
        os.makedirs('test_images')

    if args.fine_tuned_unet is not None:
        if not os.path.exists(args.res_path):
            os.makedirs(args.res_path)
    else:
        if not os.path.exists(args.res_path):
            os.makedirs(args.res_path)

    # make image directory
    if not os.path.exists(os.path.join(args.res_path, args.model_id)):
        os.makedirs(os.path.join(args.res_path, args.model_id))
        os.makedirs(os.path.join(args.res_path, args.model_id, 'moefication'))

    # make directory for model id
    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'images')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'images'))
        os.makedirs(os.path.join(args.res_path, args.model_id, 'images', 'evaluation_coco'))

    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'sparsity')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'sparsity'))

    if not os.path.exists(args.moefication['img_save_path']):
        os.makedirs(args.moefication['img_save_path'])
    
    topk_experts = args.moefication['topk_experts']
    if not os.path.exists(os.path.join(args.save_path, f'moe_images_{topk_experts}')):
        os.makedirs(os.path.join(args.save_path, f'moe_images_{topk_experts}'))

    
    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'])):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective']))

    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], 'images')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], 'images'))
    
    ratio = args.modularity['skilled_neuron_ratio']
    if not os.path.exists(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], f'skilled_neuron_ratio_{ratio}')):
        os.makedirs(os.path.join(args.res_path, args.model_id, 'modularity', args.modularity['adjective'], f'skilled_neuron_ratio_{ratio}'))
    
    if not os.path.exists(args.modularity['remove_expert_path']):
        os.makedirs(args.modularity['remove_expert_path'])
    if not os.path.exists(args.modularity['remove_expert_path_val']):
        os.makedirs(args.modularity['remove_expert_path_val'])
    if not os.path.exists(args.modularity['skill_expert_path']):
        os.makedirs(args.modularity['skill_expert_path'])
    if not os.path.exists(args.modularity['skilled_experts_t_test']):
        os.makedirs(args.modularity['skilled_experts_t_test'])
    if not os.path.exists(args.modularity['skilled_experts_moefy']):
        os.makedirs(args.modularity['skilled_experts_moefy'])

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

    elif 'xl-base-1.0' in args.model_id:
        model = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float32)

    return model, num_geglu

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
    
        # change result directory
        if self.fine_tuned_unet is not None:
            self.res_path = os.path.join(self.res_path, 'fine-tuned-relu')
        else:
            self.res_path = os.path.join(self.res_path, 'baseline')
        self.save_path = os.path.join(self.res_path, self.model_id, exp_name)

    def configure(self, exp_name):      
        if exp_name == 'modularity':  
            self.save_path = os.path.join(self.res_path, self.model_id, exp_name, self.modularity['adjective'])
        ratio = self.modularity['skilled_neuron_ratio']
        self.modularity['skill_expert_path'] = os.path.join(self.res_path, self.model_id, exp_name, self.modularity['adjective'], f'skilled_neuron_ratio_{ratio}')
        self.modularity['skilled_experts_t_test'] = os.path.join(self.res_path, self.model_id, exp_name, self.modularity['adjective'], f'skilled_experts_t_test')
        self.modularity['skilled_experts_moefy'] = os.path.join(self.res_path, self.model_id, exp_name, self.modularity['adjective'], f'skilled_experts_moefy')
        self.modularity['img_save_path'] = os.path.join(self.save_path, 'images')
        if self.modularity['condition'] == 'greater':
            self.modularity['remove_expert_path'] = os.path.join(self.modularity[f'skilled_neuron_ratio_{ratio}'], 'remove_experts')
            self.modularity['remove_expert_path_val'] = os.path.join(self.modularity[f'skilled_neuron_ratio_{ratio}'], 'remove_experts_val')
        elif self.modularity['condition'] == 't_test':
            self.modularity['remove_expert_path'] = os.path.join(self.modularity['skilled_experts_t_test'], 'remove_experts')
            self.modularity['remove_expert_path_val'] = os.path.join(self.modularity['skilled_experts_t_test'], 'remove_experts_val')
        elif self.modularity['condition'] == 'moefy':
            self.modularity['remove_expert_path'] = os.path.join(self.modularity['skilled_experts_moefy'], 'remove_experts')
            self.modularity['remove_expert_path_val'] = os.path.join(self.modularity['skilled_experts_moefy'], 'remove_experts_val')
        
    
        topk_experts = self.moefication['topk_experts']
        self.moefication['img_save_path'] = os.path.join(self.save_path, f'moe_images_{topk_experts}')
        
        print("Saving images at", {self.moefication['img_save_path']})            

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