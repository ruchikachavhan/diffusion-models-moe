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
from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights

# python benchmarks/object_erase.py  --concepts_to_remove "Golf Ball" --dataset_type modularity/datasets/imagenet_prompts.csv --fine_tuned_unet union-timesteps
checkpoints_dict = {
    'Church': 'Church_0.4.pt',
    'Golf Ball': 'Golf Ball_0.3.pt',
    'English Springer': 'English Springer_0.4.pt',
    'Garbage Truck': 'Garbage Truck_0.4.pt',
    'Chain Saw': 'Chain Saw_0.2.pt',
    'Tench': 'Tench_0.4.pt',
    'French Horn': 'French Horn_0.2.pt',
    'Parachute': 'Parachute_0.2.pt',
    'Gas Pump': 'Gas Pump_0.2.pt',
    'Cassette Player': 'Cassette Player_0.0.pt',  
}
select_ratios = {
    'golf ball': '0.4',
    'english springer': '0.6',
    'garbage truck': '0.0',
    'chain saw': '0.0',
    'tench': '0.3',
    'french horn': '0.0',
    'parachute': '0.0',
    'gas pump': '0.0',
    'cassette player': '0.0',
    'church': '0.0',
    'all_imagenette_objects': ''
}

uce_model_dict = {
    'all_imagenette_objects': 'erased-imagenette-towards_-preserve_false-sd_1_4-method_replace.pt'
}

class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.prompts = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']

        # select only prompts that have the concept to remove
        self.prompts = [(self.prompts[i], self.seeds[i], concepts_to_remove) for i in range(len(self.prompts)) if concepts_to_remove.lower() == self.labels[i].lower()]
        
        print(f"Number of prompts: {len(self.prompts)}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        prompt = prompt.replace("an image", "a photo")
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    

class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.dataset = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']
        self.dataset = [(self.dataset[i], self.seeds[i], self.labels[i].lower()) for i in range(len(self.dataset)) if concepts_to_remove.lower() != self.labels[i].lower()]

        # select only 100 prompts per class

        labels_dict = {}
        self.prompts = []
        for i in range(len(self.dataset)):
            label = self.labels[i].lower()
            if label == concepts_to_remove:
                continue
            if label not in labels_dict:
                labels_dict[label] = 1
                self.prompts.append((self.dataset[i][0], self.dataset[i][1], self.dataset[i][2]))
            elif labels_dict[label] < 100:
                labels_dict[label] += 1
                self.prompts.append((self.dataset[i][0], self.dataset[i][1], self.dataset[i][2]))
            print(labels_dict)
        print(f"Number of prompts: {len(self.prompts)}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        prompt = prompt.replace("an image", "a photo")
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
       
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
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=1, type=int, help='Batch size')

    args = args.parse_args()
    return args

def main():
    args = args_parser()

    if args.dataset_type == 'modularity/datasets/imagenette.csv':
        data = pd.read_csv(args.dataset_type)
        dataloader = torch.utils.data.DataLoader(CustomDatasetErasure(data, args.concepts_to_remove), batch_size=args.batch_size, shuffle=False)
    elif args.dataset_type == 'modularity/datasets/imagenette.csv_keep':
        data = pd.read_csv('modularity/datasets/imagenette.csv')
        dataloader = torch.utils.data.DataLoader(CustomDatasetKeep(data, args.concepts_to_remove), batch_size=args.batch_size, shuffle=False)
    # Pre-trained model
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to(args.gpu)


    if args.fine_tuned_unet is None:
        # load wanda remover for removing neurons
        path_expert_indx = os.path.join(args.root_template % (str(args.seed), args.concepts_to_remove), 'skilled_neuron_wanda', '0.01')
        print(f"Path expert index: {path_expert_indx}")
        neuron_remover = WandaRemoveNeuronsFast(seed = args.seed, path_expert_indx = path_expert_indx, 
                    T = args.timesteps, n_layers = args.n_layers, replace_fn = GEGLU, 
                    keep_nsfw =True)
        if 'imagenette.csv' in args.dataset_type:
            output_path = os.path.join('benchmarking results/unified', args.concepts_to_remove, 'keep_0.01')
        elif 'imagenette.csv_keep' in args.dataset_type:
            output_path = os.path.join('benchmarking results/unified', args.concepts_to_remove, 'erased_0.01')
    
    if args.fine_tuned_unet == 'uce':
        model_path = os.path.join('../unified-concept-editing/models/', uce_model_dict[args.concepts_to_remove])
        print(f"Loading fine-tuned UNet from {model_path}")
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
        unet.load_state_dict(torch.load(model_path))
        remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'
        remover_model = remover_model.to(args.gpu)
        if args.dataset_type.endswith('.csv'):
            output_path = os.path.join(f'benchmarking results/{args.fine_tuned_unet}', args.concepts_to_remove, 'keep no')
        elif 'imagenette.csv_keep' in args.dataset_type:
            output_path = os.path.join(f'benchmarking results/{args.fine_tuned_unet}', args.concepts_to_remove, 'erased')
    if args.fine_tuned_unet == 'union-timesteps':
        unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
        # root_template = f''
        if args.concepts_to_remove.lower() != 'all_imagenette_objects':
            best_ckpt_path = os.path.join('results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/checkpoints', args.concepts_to_remove.lower() + f"_{select_ratios[args.concepts_to_remove]}.pt")
        else:
            best_ckpt_path = os.path.join('results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/checkpoints', 'all_imagenette_objects.pt')
        print(f"Loading fine-tuned UNet from {best_ckpt_path}")
        unet.load_state_dict(torch.load(best_ckpt_path))
        remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
        output_path = f'benchmarking results/{args.fine_tuned_unet}/{args.dataset_type}/{args.concepts_to_remove}'
        remover_model = remover_model.to(args.gpu)
        if args.dataset_type.endswith('.csv'):
            output_path = os.path.join(f'benchmarking results/{args.fine_tuned_unet}', args.concepts_to_remove, 'keep no')
        elif 'imagenette.csv_keep' in args.dataset_type:
            output_path = os.path.join(f'benchmarking results/{args.fine_tuned_unet}', args.concepts_to_remove, 'erased')


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to(args.gpu)
    classifier.eval()

    preprocess = weights.transforms()

    all_gen_data = {}
    for iter, (prompt, seed, label) in enumerate(dataloader):
        # if iter > 100:
        #     break
        print(iter, prompt, seed, label)

        if not os.path.exists(os.path.join(output_path, f'{args.concepts_to_remove}_{iter}.png')):
        # if True:
            # fix the seed
            prompt = prompt[0]
            label = label[0]
            if args.fine_tuned_unet is None:
                neuron_remover.seed = seed.item()
                neuron_remover.reset_time_layer()
                # remove neurons
                removal_images, _ = neuron_remover.observe_activation(model, prompt)
            elif args.fine_tuned_unet in ['union-timesteps', 'uce']:
                # fix seed
                seed = seed[0]
                torch.manual_seed(seed)
                np.random.seed(seed)
                removal_images = remover_model(prompt).images[0]
            # save the image
            removal_images.save(os.path.join(output_path, f'{args.concepts_to_remove}_{iter}.png'))

            img = preprocess(removal_images)
            img = img.to(args.gpu).unsqueeze(0)
            with torch.no_grad():
                logits = classifier(img)
                 
            s, indices = torch.topk(logits, 1)
            indices = indices.cpu().numpy()
            pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
            print(pred_labels)

            all_gen_data[f'{args.concepts_to_remove}_{iter}'] = (removal_images, label)
        else:
            print(f"Skipping {args.concepts_to_remove}_{args.concepts_to_remove}_{iter}.png")
            try:
                all_gen_data[f'{args.concepts_to_remove}_{iter}'] = (
                    Image.open(os.path.join(output_path, f'{args.concepts_to_remove}_{iter}.png')), 
                    label[0])
            except:
                continue
    
    print("Done")

    print("Calculating pre-trained ResNet accuracy")
    all_categories = weights.meta["categories"]
    all_categories = [c.lower() for c in all_categories]
    # print(all_categories)
    avg_acc = 0
    num_valid = 0
    for key, value in all_gen_data.items():
        img = value[0]
        label = value[1].strip()
        img = preprocess(img)
        img = img.to(args.gpu).unsqueeze(0)
        with torch.no_grad():
            logits = classifier(img)
        # calculate top-5 accuracy
        s, indices = torch.topk(logits, 5)
        
        indices = indices.cpu().numpy()
        pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
        # if one word of labels is in the predicted labels, then it is correct
        # first check if true label is present in all_categories
        # label_words = label.split(" ")
        # check if there is hyphen or a _ un labek and replace it with " "
        # print("Label before", label, pred_labels)
        # valid = True if label in all_categories else False
        # if valid:

        for pred in pred_labels:
            label_words = label.split(" ")
            pred_words = pred.split(" ")
            found = False
            for w in pred_words:
                # print(w, label_words)
                if w in label_words:
                    avg_acc +=1 
                    found = True
                    break
            if found:
                break


            # if label.lower() in pred.lower():
            #     avg_acc += 1
            #     break
            # else:
            #     print(label, pred, key, pred_labels)
                    
    # print("num_valid", num_valid)
    # print(f"Average accuracy: {avg_acc/num_valid}")

    results = {}
    results['acc'] = avg_acc/len(all_gen_data)
    print(results)

    # save results
    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump(results, f)

    

if __name__ == '__main__':
    main()