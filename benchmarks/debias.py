import json
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter
sys.path.append('modularity')
from mod_utils import get_prompts, LLAVAScorer
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from diffusers.models.activations import LoRACompatibleLinear
from neuron_receivers import RemoveExperts, RemoveNeurons, WandaRemoveNeurons, WandaRemoveNeuronsFast
sys.path.append('moefication')
from helper import modify_ffn_to_experts
from PIL import ImageDraw, ImageFont
from diffusers.models.activations import GEGLU
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
 
# if msfw is on, blur the image a 100 times
def blur_image(image, is_nsfw):
    if is_nsfw:
        for i in range(100):
            image = image.filter(ImageFilter.BLUR)
    return image

def remove_experts(adj_prompts, model, neuron_receiver, args, bounding_box, save_path_root, base_prompts=None):
    
    # run experiment on multiple seeds
    seeds = np.arange(31, 250, 1)
    for seed in seeds:
        save_path = os.path.join(save_path_root, f'seed_{seed}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # fix seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        neuron_receiver.seed = seed
        iter = 0
        for ann_adj in adj_prompts:
            if iter >= 2 and args.dbg:
                break
            print("text: ", ann_adj)
            # fix seed
            torch.manual_seed(seed)
            np.random.seed(seed)
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
            # font = ImageFont.load_default(size=15)
            draw.text((80, 15), ann_adj, (255, 255, 255))
            draw.text((350, 15), 'w/o experts', (255, 255, 255))

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
    
    # # change ratio to value of conf_val from hparams
    adjectives = args.modularity['adjective']
    base_prompts, adj_prompts, _ = get_prompts(args)
    # conf_val = hparams['conf_val']
    all_timesteps = False
    
    hparams = None
    hpo_method = None

    
    if args.fine_tuned_unet is None:
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
            func = WandaRemoveNeuronsFast
        neuron_receiver =  func(seed=args.seed, path_expert_indx = args.modularity['skill_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['skill_neuron_path'],
                                T=args.timesteps, n_layers=num_geglu, replace_fn=replace_fn, keep_nsfw=args.modularity['keep_nsfw'])
                                            
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
        
        output_path = 'benchmarking results/unified/debiasing_' + args.modularity['adjective']
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # remove experts
        remove_experts(adj_prompts, model, neuron_receiver, args, 
                    bounding_box=bb_coordinates_layer_adj if args.modularity['bounding_box'] else None, 
                    save_path_root=output_path,
                        base_prompts=base_prompts)
        
    if args.fine_tuned_unet == 'union-timesteps':
        args.concepts_to_remove = args.modularity['adjective']
        args.dataset_type = args.modularity['file_name']
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
        root_template = f'results/results_seed_%s/stable-diffusion/baseline/{args.model_id}'
        best_ckpt_path = os.path.join(root_template % (str(args.seed)), 'checkpoints', f'{args.concepts_to_remove}_0.4.pt')
        print(f"Loading fine-tuned UNet from {best_ckpt_path}")
        unet.load_state_dict(torch.load(best_ckpt_path))
        remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'
        remover_model = remover_model.to(args.gpu)

        model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        model = model.to('cuda:0')

        # read file
        with open(os.path.join('modularity/datasets', args.modularity['file_name']+'.txt'), 'r') as f:
            objects = f.readlines()

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # remove experts
        
        seeds = np.arange(66, 250, 1)
        for seed in seeds:
            save_path = os.path.join(output_path, f'seed_{seed}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for obj in objects:
                obj = obj.strip()
                prompt = f'a photo of a {obj.lower()}'
                print(f"Object: {prompt}", "Seed: ", seed)
                # run model for the original text
                torch.manual_seed(seed)
                np.random.seed(seed)
                out = remover_model(prompt).images[0]
                # save the image
                out.save(os.path.join(save_path, f'{args.concepts_to_remove}_{obj}.png'))

                # original gender
                torch.manual_seed(seed)
                np.random.seed(seed)
                image = model(prompt).images[0]
                image.save(os.path.join(save_path, f'{args.concepts_to_remove}_{obj}_original.jpg'))
                print("Image saved in ", save_path)
            



if __name__ == "__main__":
    main()