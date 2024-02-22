import os
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from relufy_model import find_and_change_geglu

def make_dirs(args):
    if args.fine_tuned_unet is not None:
        args.res_path = os.path.join(args.res_path, 'fine-tuned-relu')
        if not os.path.exists(args.res_path):
            os.makedirs(args.res_path)
    # make image directory
    if not os.path.exists(os.path.join(args.res_path, 'images')):
        os.makedirs(os.path.join(args.res_path, 'images'))
    # make sparsity directory
    if not os.path.exists(os.path.join(args.res_path, 'sparsity')):
        os.makedirs(os.path.join(args.res_path, 'sparsity'))
    
    # make directory for model id
    if not os.path.exists(os.path.join(args.res_path, 'images', args.model_id)):
        os.makedirs(os.path.join(args.res_path, 'images', args.model_id))
    if not os.path.exists(os.path.join(args.res_path, 'sparsity', args.model_id)):
        os.makedirs(os.path.join(args.res_path, 'sparsity', args.model_id))

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
            model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
        num_geglu = 16

    elif 'xl-base-1.0' in args.model_id:
        model = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float32)
        num_geglu = 70

    return model, num_geglu