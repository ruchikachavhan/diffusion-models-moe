import os
import numpy as np
import torch
import json
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from relufy_model import find_and_change_geglu

def make_dirs(args):
    if args.fine_tuned_unet is not None:
        args.res_path = os.path.join(args.res_path, 'fine-tuned-relu')
        if not os.path.exists(args.res_path):
            os.makedirs(args.res_path)
    else:
        args.res_path = os.path.join(args.res_path, 'baseline')
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
        num_geglu = 16

    elif 'xl-base-1.0' in args.model_id:
        model = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float32)
        num_geglu = 70

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