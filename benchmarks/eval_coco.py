import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
from torchvision import transforms
sys.path.append(os.getcwd())
import utils
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from neuron_receivers import WandaRemoveNeuronsFast
import pandas as pd
import argparse
import tqdm
from diffusers.models.activations import GEGLU, GELU
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

def calculate_fid(original_images, generated_images):
    fid = FID()
    fid.update(original_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()


things = ['cat', 'dog', 'bird', 'car', 'bear', 'bat', 'ball', 'apple', 'banana', 'house', 'tree', 'flower', 'room', 'building', 'wall', 'chicken', 'duck']
humans = ['man', 'woman', 'child', 'boy', 'girl', 'kid', 'teacher', 'nurse', 'doctor', 'professor', 'guy', 'couple']
wanda_thr = {
    'Alex Alemany,painter': 0.05,
    'John Howe,illustrator': 0.05,
    'Antonio J. Manzanedo': 0.05,
    'Alpo Jaakola,painter': 0.05,
    'Abraham Mintchine,painter': 0.05,
    'Apollinary Vasnetsov,landscape art,history painting': 0.05,
    'John Constable,realism,landscape art': 0.05,
    'Johannes Vermeer,genre painting,portrait': 0.05,
    'Nicolas Mignard,mythological painting,painter': 0.05,
    'John Whitcomb': 0.05,
    'Amedeo Modigliani,landscape art,portrait': 0.05,
    'Jordan Grimmer': 0.05,
    'A.J.Casson': 0.05,
    'Akira Toriyama,mangaka,fantasy': 0.05,
    'Salvador Dali': 0.05,
    'Greg Rutkowski': 0.05,
    'Jeremy Mann': 0.05,
    'Van Gogh': 0.02,
    'Monet': 0.05,
    'Pablo Picasso': 0.05,
    'naked': 0.01,
    'memorize': 0.01, 
    '5artists_naked': '',
    '5artists': 0.95,
    '10artists': 0.95,
    'Juliana Huxtable,Valerie Hegarty,Wendy Froud,Kobayashi Kiyochika,Paul Laffoley': 0.02
}
uce_models_dict = {
    '50artists': 'erased-50artists-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '1artist': 'erased-algernon blackwood-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '10artists': 'erased-asger jorn_eric fischl_johannes vermeer_apollinary vasnetsov_naoki urasawa_nicolas mignard_john whitcomb_john constable_warwick globe_albert marquet-towards_art-preserve_true-sd_1_4-method_replace.pt',
    '5artists': 'erased-juliana huxtable_valerie hegarty_wendy froud_kobayashi kiyochika_paul laffoley-towards_art-preserve_true-sd_1_4-method_replace.pt',
}
# make dataloader for coco
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, imgs, anns, transform):
        self.imgs = imgs
        self.anns = anns
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        ann = self.anns[idx]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        return img, ann 
        
class HoldoutDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, transform):
        self.prompts = prompts
        self.transform = transform

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return prompt, prompt
    
def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    args.add_argument('--seed', type=int, default=0, help='seed')
    args.add_argument('--replace_fn', type=str, default='GEGLU', help='replace function')
    args.add_argument('--keep_nsfw', type=bool, default=True, help='keep nsfw')
    args.add_argument('--dbg', action='store_true', help='debug')
    args.add_argument('--gpu', type=int, default=7, help='gpu')
    args.add_argument('--n_layers', type=int, default=16, help='n layers')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', default=None, help='List of concepts to remove')
    args.add_argument('--dataset_type', default=None, help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/art/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=4, type=int, help='Batch size')

    args = args.parse_args()
    return args

def main():
    args = args_parser()
    # Step 1 - Read thhe COCO dataset

    if args.dataset_type == 'coco':
        imgs, anns = utils.coco_dataset('../COCO-vqa', 'val', 30000)
        print("Evaluating on COCO dataset", len(imgs), len(anns))
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # make a dataloadet for coco
        dataloader = torch.utils.data.DataLoader(COCODataset(imgs, anns, transform), batch_size=args.batch_size, shuffle=False)
    elif args.dataset_type == 'holdout':
        print("Testing model on Holdout dataset of artists")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        # read csv file with holdout artists
        data = pd.read_csv('modularity/datasets/holdout100_prompts.csv')
        # create a list of artsist from data
        prompts = data['prompt'].tolist()
        dataloader = torch.utils.data.DataLoader(HoldoutDataset(prompts, transform), batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError("Dataset type not found")
    
    # Pre-trained model
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to(args.gpu)

    if args.fine_tuned_unet == 'uce':
        # load a baseline model and fine tune it
        unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(os.path.join('../unified-concept-editing/models', uce_models_dict[args.concepts_to_remove])))
        remover_model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', unet=unet, torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'

    if args.fine_tuned_unet == 'concept-ablation':
        remover_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        remover_model = remover_model.to(args.gpu)
        model_path = os.path.join('../concept-ablation/diffusers', 'logs_ablation', args.concepts_to_remove, 'delta.bin')
        model_ckpt = torch.load(model_path)
        if 'text_encoder' in model_ckpt:
            remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
        for name, params in remover_model.unet.named_parameters():
            if name in model_ckpt['unet']:
                params.data.copy_(model_ckpt['unet'][f'{name}'])
        # remover_model.load_model(os.path.join('../concept-ablation/diffusers', 'logs_ablation', args.concepts_to_remove, 'delta.bin'))
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'

    if args.fine_tuned_unet is None:
        # initalise Wanda neuron remover
        path_expert_indx = os.path.join(args.root_template % (str(args.seed), args.concepts_to_remove), 'skilled_neuron_wanda', str(wanda_thr[args.concepts_to_remove]))
        print(f"Path expert index: {path_expert_indx}")
        neuron_remover = WandaRemoveNeuronsFast(seed = args.seed, path_expert_indx = path_expert_indx, T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, keep_nsfw =True, remove_timesteps=20)
        output_path = f'benchmarking results/unified/{args.dataset_type}/{args.concepts_to_remove}'


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # test model on dataloader
    for iter, (img, prompt) in enumerate(dataloader):

        # check if image is present in putput path
        if os.path.exists(os.path.join(output_path, f"original_{iter * args.batch_size}.png")):
            print(f"Skipping iteration {iter}")
            continue

        print("Iteration number", iter, prompt)
        prompt = [p for p in prompt]

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        gen_images = model(prompt=prompt, safety_checker=safety_checker_).images
        # save images 

        # remove neurons
        if args.concepts_to_remove is not None:
            if args.fine_tuned_unet in ['uce', 'concept-ablation']:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                removal_images = remover_model(prompt=prompt, safety_checker=safety_checker_).images
            elif args.fine_tuned_unet is None:
                neuron_remover.reset_time_layer()
                removal_images, _ = neuron_remover.observe_activation(model, prompt)

        # stitch images and save
        for i, image in enumerate(gen_images):
            # num_images observed till now
            num_images = iter * args.batch_size + i
            image.save(os.path.join(output_path, f"sd_{num_images}.png"))
            # save original image
            if args.dataset_type == 'coco':
                # only save image for coco dataset
                save_image = img[i] * 255
                save_image = save_image.to(torch.uint8)
                save_image = transforms.ToPILImage()(save_image)
                save_image.save(os.path.join(output_path, f"original_{num_images}.png"))

            if args.concepts_to_remove is not None:
                removal_images[i].save(os.path.join(output_path, f"removed_{num_images}.png"))
        


    # Get concept checkets
    # memorized_checker = MemorizedPromptChecker(device=args.gpu, objects=things, neg_objects=None)
    # nudity_checker = NudityChecker(device=args.gpu, objects=humans, neg_objects=things)
    # art_style_checker = ArtStyleChecker(device=args.gpu, objects=things, neg_objects=humans)

    # # initialise neuron receivers
    # all_concepts_to_remove = []
    # if 'naked' in args.concepts_to_remove:
    #     all_concepts_to_remove += ['naked']
    # if 'art' in args.concepts_to_remove:
    #     all_concepts_to_remove += art_styles
    # if 'memorize' in args.concepts_to_remove:
    #     all_concepts_to_remove += ['memorize']
    
    # # SD model
    # model, num_geglu, replace_fn = utils.get_sd_model(args)
    # model = model.to(args.gpu)

    # clip_image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # clip_image_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    
    # neuron_remover = MultiConceptRemoverWanda(root = args.root_template, seed = args.seed, 
    #         T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, 
    #         keep_nsfw = args.keep_nsfw, concepts_to_remove = all_concepts_to_remove, wanda_thr=wanda_thr)

    # output_path = os.path.join('benchmarking results', 'unified', 'coco')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # results = {}

    # original_images, removal_images = [], []
    # score = []
    # iter = 0
    # for img, prompt in tqdm.tqdm(zip(imgs, anns)):
    #     if args.dbg and iter > 5:
    #         break
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     orig_image = Image.open(os.path.join('../COCO-vqa', img))
    #     generated_image = model(prompt=prompt).images[0]
    #     original_images.append(transform(Image.open(os.path.join('../COCO-vqa', img)))) # original image
    #     # find concpet if present
    #     remove_nudity = nudity_checker.decide(prompt)
    #     remove_art_style = art_style_checker.decide(prompt)
    #     remove_memorized = memorized_checker.decide(prompt)
    #     pred = [remove_nudity, remove_art_style]
    #     concept_remove = []
    #     if remove_nudity == 'naked' and 'naked' in all_concepts_to_remove:
    #         concept_remove.append('naked')
    #     if remove_art_style != 'none' and remove_art_style in all_concepts_to_remove:
    #         concept_remove.append(remove_art_style)
    #     if remove_memorized == 'memorize' and 'memorize' in all_concepts_to_remove:
    #         concept_remove.append('memorize')

    #     # remove neurons if concept is detected
    #     print(f'Prompt: {prompt}', f'Prediction: {concept_remove}')
    #     results[prompt] = {}
    #     results[prompt]['pred'] = concept_remove
    #     # check if concepts need to be removed
    #     if len(concept_remove) > 0:
    #         output_after_removal, single_image_removal = neuron_remover.remove_concepts(model, prompt, concept_remove)
    #         name = '_'.join(concept_remove)
    #         output_after_removal.save(os.path.join(output_path, f'{iter}_{name}.png'))
    #         removal_images.append(transform(output_after_removal))
    #     else:
    #         output_after_removal = generated_image
    #         removal_images.append(transform(generated_image))
        
    #     # clip score
    #     inputs = clip_image_processor(images = orig_image, return_tensors = 'pt', padding = True)
    #     orig_features = clip_image_encoder.get_image_features(**inputs)

    #     inputs = clip_image_processor(images = output_after_removal, return_tensors = 'pt', padding = True)
    #     removal_features = clip_image_encoder.get_image_features(**inputs)

    #     print(f"Original features: {orig_features.shape}, Removal features: {removal_features.shape}")

    #     similarity = torch.nn.functional.cosine_similarity(orig_features, removal_features, dim = -1)
    #     print(similarity.item())
    #     score.append(similarity.item())
    #     iter += 1

    # original_images = torch.stack(original_images) * 255
    # removal_images = torch.stack(removal_images) * 255
    # # convert to unit8
    # original_images = original_images.to(torch.uint8)
    # removal_images = removal_images.to(torch.uint8)
    # fid = calculate_fid(original_images, removal_images)
    # score = np.mean(score)

    # print(f"FID: {fid}, Score: {score}")
    # results['fid'] = fid
    # results['score'] = score
    # # write to file
    # with open(os.path.join(output_path, 'results.json'), 'w') as f:
    #     json.dump(results, f)

if __name__ == '__main__':
    main()