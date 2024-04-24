import json
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
from mod_utils import get_prompts, LLAVAScorer
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from neuron_receivers import RemoveExperts, RemoveNeurons, WandaRemoveNeurons
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from PIL import ImageDraw, ImageFont
 
# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, neuron_receiver, args, bounding_box, save_path, base_prompts=None):
    iter = 0

    for ann_adj in adj_prompts:
        if iter >= 2 and args.dbg:
            break
        print("text: ", ann_adj)
        # fix seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # run model for the original text
        # out = model(ann_adj).images[0]
        if 'lcm' in args.model_id:
            out = model(ann_adj, num_inference_steps=4, guidance_scale=8.0).images[0]
        else:
            out = model(ann_adj).images[0]

        neuron_receiver.reset_time_layer()
        # ann_adj = ann_adj + '\n'
        out_adj, _ = neuron_receiver.observe_activation(model, ann_adj,
                                                         bboxes=bounding_box[ann_adj] if bounding_box is not None else None)


        # stitch the images to keep them side by side
        out = out.resize((256, 256))
        out_adj = out_adj.resize((256, 256))
        # make bigger image to keep both images side by side with white space in between
        new_im = Image.new('RGB', (530, 290))

        if args.modularity['keep_nsfw']:
            out = blur_image(out, args.modularity['condition']['is_nsfw'])
            
        new_im.paste(out, (0,40))
        new_im.paste(out_adj, (275,40))

        # write the prompt on the image
        draw = ImageDraw.Draw(new_im)
        font = ImageFont.load_default(size=15)
        draw.text((80, 15), ann_adj, (255, 255, 255), font=font)
        draw.text((350, 15), 'w/o experts', (255, 255, 255), font=font)

        # obj_name = base_prompts[iter].split(' ')[-1] if base_prompts is not None else ann_adj
        # obj_name = base_prompts[iter] if base_prompts is not None else ann_adj
        obj_name = ann_adj

        new_im.save(os.path.join(save_path, f'img_{iter}_{obj_name}.jpg'))

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
    # read hyperparameters saved
    adjective = args.modularity['adjective']
    # hpo_method = 'noise_hpo_iterations'
    # if hpo_method == 'noise_hpo_iterations':
    #     hparams = json.load(open(f'modularity/hpo_results/{adjective}.json', 'r'))
    # elif hpo_method == 'llava':
    #     hparams = json.load(open(f'modularity/hpo_results_llava/{adjective}_hpo.json', 'r'))
    
    # # change ratio to value of conf_val from hparams
    adjectives = args.modularity['adjective']
    base_prompts, adj_prompts, _ = get_prompts(args)
    # conf_val = hparams['conf_val']
    all_timesteps = False
    
    hparams = None
    hpo_method = None

    # args.modularity['condition']['skill_ratio'] = str(args.modularity['condition']['skill_ratio']) + "/" + f'dof_{len(base_prompts) - 1}_conf_{conf_val}'
    # if all_timesteps:
    #     # change all timesteps to 1
    #     for key in hparams.keys():
    #         if 'conf' not in key:
    #             hparams[key] = 1
    # print(hparams)

    args.configure('modularity')
    if hpo_method is not None:
        args.modularity['remove_neuron_path'] = os.path.join(args.modularity['remove_neuron_path'], hpo_method)
        args.modularity['remove_neuron_path_val'] = os.path.join(args.modularity['remove_neuron_path_val'], hpo_method)
    if all_timesteps:
        # change remove_neurons_path
        args.modularity['remove_neuron_path'] = os.path.join(args.modularity['remove_neuron_path'], 'all_timesteps')
        args.modularity['remove_neuron_path_val'] = os.path.join(args.modularity['remove_neuron_path_val'], 'all_timesteps')
    if not os.path.exists(args.modularity['remove_neuron_path']):
        os.makedirs(args.modularity['remove_neuron_path'])
    if not os.path.exists(args.modularity['remove_neuron_path_val']):
        os.makedirs(args.modularity['remove_neuron_path_val'])

    # model 
    model, num_geglu, replace_fn = utils.get_sd_model(args)
    args.replace_fn = replace_fn
    model = model.to(args.gpu)

    # Neuron receiver with forward hooks
    
    if args.modularity['condition']['name'] == 't_test':
        func = RemoveNeurons if args.modularity['condition']['remove_neurons'] else RemoveExperts
    elif args.modularity['condition']['name'] == 'wanda':
        func = WandaRemoveNeurons
    neuron_receiver =  func(seed=args.seed, path_expert_indx = args.modularity['skill_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['skill_neuron_path'],
                            T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.modularity['keep_nsfw'], 
                            remove_timesteps = hparams)
                                           
    # adjectives = args.modularity['adjective']
    # base_prompts, adj_prompts, _ = get_prompts(args)
    
    # read file
    with open(os.path.join('modularity/datasets', args.modularity['file_name']+'.txt'), 'r') as f:
        objects = f.readlines()
    objects = [obj.strip() for obj in objects]

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
                     base_prompts=base_prompts)

    # read val_dataset
    if not os.path.exists(f'modularity/datasets/val_things_{adjective}.txt'):
        print(f"Validation dataset not found for {adjective}")
        return
    
    with open(f'modularity/datasets/val_things_{adjective}.txt') as f:
        val_objects = f.readlines()
    
    val_base_prompts = [f'{thing.strip()}' for thing in val_objects]
    
    # remove experts from val_dataset
    remove_experts(val_base_prompts, model, neuron_receiver, args, 
                   bounding_box=args.modularity['bounding_box'] if args.modularity['bounding_box'] else None,
                    save_path=args.modularity['remove_expert_path_val'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path_val'], 
                    base_prompts=None)
    

if __name__ == "__main__":
    main()