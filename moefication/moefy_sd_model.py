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
import moe_utils
from helper import make_templates, test_template, get_model_block_config
sys.path.append(os.getcwd())
import utils
from diffusers.models.activations import GEGLU

            
def main():
    args = utils.Config('experiments/moefy_config.yaml', 'moefication')
    args.configure('moefication')

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)
    block_config = get_model_block_config(args.model_id)

    # Save the model as a .pt file, MOE code base doe sthis
    torch.save(model.unet.state_dict(), os.path.join(args.save_path, 'model.pt'))
    # make MOE config 
    config = moe_utils.ModelConfig(os.path.join(args.save_path, 'model.pt'), args.res_path, split_size=args.moefication['expert_size'])

    # Templates for FFN name keys
    templates = make_templates(args.moefication['templates'], block_config)
    test_template(templates, model)

    for template in templates:
        print(f"Splitting parameters for {template}")
        # For every FFN, we now split the weights into clusters by using KMeansConstrained
        split = moe_utils.ParamSplit(config, template)
        split.split()
        split.cnt()
        split.save()


if __name__ == "__main__":
    main()



