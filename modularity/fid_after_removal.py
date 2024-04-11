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
from torchvision import transforms
sys.path.append('sparsity')
from eval_coco import calculate_fid
 
# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, neuron_receiver, args, bounding_box, save_path, base_prompts=None, remove_token_idx=None):
    iter = 0
    neuron_receiver.remove_token_idx = remove_token_idx
    base_images, concept_images, after_removal_images = [], [], []
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    for ann_base, ann_adj in zip(base_prompts, adj_prompts):
        if iter >= 4 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # run model for the original text
        # out = model(ann_adj).images[0]
        if 'lcm' in args.model_id:
            out_base = model(ann_base, num_inference_steps=4, guidance_scale=8.0).images[0]
            out = model(ann_adj, num_inference_steps=4, guidance_scale=8.0).images[0]
        else:
            out_base = model(ann_base).images[0]
            out = model(ann_adj).images[0]

        # out.save(os.path.join('test_images', f'img_original.jpg'))

        neuron_receiver.reset_layer()
        # ann_adj = ann_adj + '\n'
        out_adj, _ = neuron_receiver.observe_activation(model, ann_adj, 
                            bboxes=bounding_box[ann_adj] if bounding_box is not None else None)
        
        base_images.append(transform(out_base.convert('RGB')))
        after_removal_images.append(transform(out_adj.convert('RGB')))
        concept_images.append(transform(out.convert('RGB')))
        iter += 1

    # calculate FID
    base_images = torch.stack(base_images) * 255
    after_removal_images = torch.stack(after_removal_images) * 255
    concept_images = torch.stack(concept_images) * 255
    fid_1 = calculate_fid(base_images, after_removal_images)
    print(f"FID score: {fid_1}")
    fid_2 = calculate_fid(concept_images, after_removal_images)
    print(f"FID score: {fid_2}")

    # save the fid scores
    with open(os.path.join(save_path, 'fid_scores.txt'), 'w') as f:
        f.write(f"FID score between base images and after removal images: {fid_1}\n")
        f.write(f"FID score between concept images and after removal images:: {fid_2}\n")

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')

    # model 
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks
   
    func = RemoveNeurons if args.modularity['condition']['remove_neurons'] else RemoveExperts
    neuron_receiver =  func(seed=args.seed, path_expert_indx = args.modularity['skill_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['skill_neuron_path'],
                            T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.modularity['keep_nsfw'])
                                           
    adjectives = args.modularity['adjective']
    base_prompts, adj_prompts, _, remove_token_idx = get_prompts(args)

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
    # args.moefication['topk_experts'] = 1.0
    # model, _, _ = modify_ffn_to_experts(model, args)
    
    # remove experts
    remove_experts(adj_prompts, model, neuron_receiver, args, 
                            bounding_box=bb_coordinates_layer_adj if args.modularity['bounding_box'] else None, 
                            save_path=args.modularity['remove_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path'], 
                            base_prompts=base_prompts, remove_token_idx=remove_token_idx)


if __name__ == "__main__":
    main()