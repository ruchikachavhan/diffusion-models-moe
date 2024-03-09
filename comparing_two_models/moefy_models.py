import os
import sys
import yaml
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision import transforms
sys.path.append(os.getcwd())
from utils import get_sd_model, coco_dataset
from eval_coco import calculate_fid
from neuron_receivers import MOEFy
sys.path.append('moefication')
import moe_utils
from helper import make_templates, test_template, get_model_block_config, modify_ffn_to_experts
sys.path.append('comparing_two_models')
from config_utils import Config

def moefy_model(model, args, name):
    block_config = get_model_block_config(args.model_id)
    # Save the model as a .pt file, MOE code base doe this

    torch.save(model.unet.state_dict(), os.path.join(args.save_path, name, f'model.pt'))
    # make MOE config 
    config = moe_utils.ModelConfig(os.path.join(args.save_path, name, f'model.pt'),
                                   os.path.join(args.save_path, name),
                                    split_size=args.moefication['expert_size'])

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

def evaluate_on_coco(model, moefier, imgs, anns, args, model_name):
    if not os.path.exists(os.path.join(args.save_path, model_name, 'coco_eval')):
        os.makedirs(os.path.join(args.save_path, model_name, 'coco_eval'))

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    orig_imgs, non_moe_imgs, moe_imgs = [], [], []

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 10 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # With MOEfication
        out_moe, _ = moefier.observe_activation(model, ann)
        # save images
        if iter < 10:
            out_moe.save(os.path.join(args.save_path, model_name, 'coco_eval', f'moe_{iter}.png'))
        iter += 1

        # Collect images
        orig_imgs.append(transform(Image.open(img).convert('RGB')))
        moe_imgs.append(transform(out_moe.convert('RGB')))

    orig_imgs = torch.stack(orig_imgs) * 255
    moe_imgs = torch.stack(moe_imgs) * 255
    fid = calculate_fid(orig_imgs, moe_imgs)
    return fid

def main():

    args1 = Config('experiments/compare_two_models.yaml', 'compare_peft')  
    args2 = Config('experiments/compare_two_models.yaml', 'compare_peft')  
    args = [args1, args2]

    # load the two models
    # number of models to compare
    model_paths = args[0].fine_tuned_unet.split(',')
    model_names = args[0].model_names.split(',')
    print(f"Model paths: {model_paths}", f"Model names: {model_names}")
    models = []

    for i, model_path in enumerate(model_paths):
        args[i].fine_tuned_unet = model_path
        args[i].model_name = model_names[i]
        model, num_geglu = get_sd_model(args[i])
        model = model.to(args[i].gpu)
        models.append(model)
        if not os.path.exists(os.path.join(args[i].save_path, model_names[i], 'param_split')):
            os.makedirs(os.path.join(args[i].save_path, model_names[i], 'param_split'))

    # load the dataset
    imgs, anns = coco_dataset(args[0].dataset['path'], 'val', args[0].inference['num_images'])  

    # moefy the models
    print(args[0].save_path, model_names[0])
    if len(os.listdir(os.path.join(args[0].save_path, model_names[0], 'param_split'))) == 0:
        for i, model in enumerate(models):
            moefy_model(model, args[i], model_names[i])
    else:
        print("Models already moefied")
    
    # # read convert ffns to experts
    for i, model in enumerate(models):
        args[i].res_path = os.path.join(args[i].res_path, model_names[i])
        model, _, _ = modify_ffn_to_experts(model, args[i])
        models[i] = model
    
    moefier = MOEFy(seed = args[0].seed)

    # # evaluate the models
    fids = []
    for i, model in enumerate(models):
        fids.append(evaluate_on_coco(model, moefier, imgs, anns, args[i], model_names[i]))

    print(f"FID scores: {fids}")
    
        
    


if __name__ == "__main__":
    main()
