import json
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
from mod_utils import get_prompts
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import RemoveExperts, RemoveNeurons
sys.path.append('moefication')
from helper import modify_ffn_to_experts
 
# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, neuron_receiver, args, bounding_box, save_path):
    iter = 0
    for ann_adj in adj_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # run model for the original text
        out = model(ann_adj).images[0]

        neuron_receiver.reset_time_layer()
        # ann_adj = ann_adj + '\n'
        out_adj, _ = neuron_receiver.observe_activation(model, ann_adj, bboxes=bounding_box[ann_adj] if bounding_box is not None else None)
        # save images
        print("Image saved in ", save_path)
        if args.modularity['keep_nsfw']:
            out = blur_image(out, args.modularity['condition']['is_nsfw'])
            blurred_out_adj = blur_image(out_adj, args.modularity['condition']['is_nsfw'])
            blurred_out_adj.save(os.path.join(save_path, f'img_{iter}_adj_blurred.jpg'))
        out.save(os.path.join(save_path, f'img_{iter}.jpg'))
        out_adj.save(os.path.join(save_path, f'img_{iter}_adj.jpg'))
        iter += 1

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')

    # model 
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks
   
    func = RemoveNeurons if args.modularity['condition']['remove_neurons'] else RemoveExperts
    neuron_receiver =  func(seed=args.seed, path_expert_indx = args.modularity['skill_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['skill_neuron_path'],
                            T=args.timesteps, n_layers=num_geglu, keep_nsfw=args.modularity['keep_nsfw'])
                                           
    adjectives = args.modularity['adjective']
    _, adj_prompts, _ = get_prompts(args)

    if args.modularity['bounding_box']:
        # read bounding box coordinates
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_adj.json')) as f:
            bb_coordinates_layer_adj = json.load(f)
            print(bb_coordinates_layer_adj.keys())
        with open(os.path.join(args.save_path, 'bb_coordinates_layer_base.json')) as f:
            bb_coordinates_layer_base = json.load(f)
    # COnvert FFns into moe
    # set args.moefication['topk_experts'] = 1 to keep all the experts  
    # so that all experts are being used in the model and the ones removed are the skilled ones
    args.moefication['topk_experts'] = 1.0
    model, _, _ = modify_ffn_to_experts(model, args)
    
    # remove experts
    remove_experts(adj_prompts, model, neuron_receiver, args, 
                   bounding_box=bb_coordinates_layer_adj if args.modularity['bounding_box'] else None, 
                   save_path=args.modularity['remove_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path'])

    # read val_dataset
    if not os.path.exists(f'modularity/datasets/val_things_{adjectives}.txt'):
        print(f"Validation dataset not found for {adjectives}")
        return
    
    with open(f'modularity/datasets/val_things_{adjectives}.txt') as f:
        val_objects = f.readlines()
    
    val_base_prompts = [f'{thing.strip()}' for thing in val_objects]
    
    # remove experts from val_dataset
    remove_experts(val_base_prompts, model, neuron_receiver, args, 
                   bounding_box=args.modularity['bounding_box'] if args.modularity['bounding_box'] else None,
                    save_path=args.modularity['remove_expert_path_val'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path_val'])
    

if __name__ == "__main__":
    main()