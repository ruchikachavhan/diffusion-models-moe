import json
import os
import sys
import torch
import tqdm
import types
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import MOEFy

def main():
    args = utils.Config('experiments/moefy_config.yaml', 'moefication')
    args.configure('moefication')

    # Input topk experts for evaluation
    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication['topk_experts'] = topk
        args.configure('moefication')
    print(f"Topk experts: {args.moefication['topk_experts']}")

    # Model
    model, _ = utils.get_sd_model(args)
    model = model.to(args.gpu)
    
    # Change FFNS to a mixture of experts
    model, _, _ = modify_ffn_to_experts(model, args)
    
    # Eval dataset
    imgs, anns = utils.coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    # MOEFIER
    moefier = MOEFy(seed = args.seed)
    moefier.test(model, relu_condition=args.fine_tuned_unet is not None)
    
    orig_imgs, non_moe_imgs, moe_imgs = [], [], []
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 10 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Without MOEfication
        out = model(ann).images[0]
        # With MOEfication
        out_moe, _ = moefier.observe_activation(model, ann)
        # Collect images
        orig_imgs.append(transform(Image.open(img).convert('RGB')))
        non_moe_imgs.append(transform(out.convert('RGB')))
        moe_imgs.append(transform(out_moe.convert('RGB')))
        iter += 1

    orig_imgs = torch.stack(orig_imgs) * 255
    non_moe_imgs = torch.stack(non_moe_imgs) * 255
    moe_imgs = torch.stack(moe_imgs) * 255
    fid = ec.calculate_fid(orig_imgs, non_moe_imgs)
    fid_moe = ec.calculate_fid(orig_imgs, moe_imgs)
    print(f"FID: {fid}, FID MOE: {fid_moe}")
    # save the fid scores
    topk_experts = args.moefication['topk_experts']
    with open(os.path.join(args.save_path, f'fid_{topk_experts}.txt'), 'w') as f:
        f.write(f"FID: {fid}, FID MOE: {fid_moe}")


if __name__ == "__main__":
    main()
