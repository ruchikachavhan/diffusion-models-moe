import os
import torch
import argparse
import json
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoProcessor
from transformers import pipeline

labels_dict = {
    'Van Gogh': 'vincent-van-gogh',
    'Monet': 'claude-monet',
    'Pablo Picasso': 'pablo-picasso',
    'Leonardo Da Vinci': 'leonardo-da-vinci',
    'Salvador Dali': 'salvador-dali',
    'Rembrandt': 'rembrandt',
}

top_k = {
    'Van Gogh': 3,
    'Monet': 3,
    'Pablo Picasso': 3,
    'Leonardo Da Vinci': 3,
    'Salvador Dali': 1,
    'Rembrandt': 3,
}

def get_labels(results, labels):
    labels = labels_dict[labels]
    keys = [r['label'] for r in results]
    pred = 1 if labels in keys else 0
    return pred

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dbg', action='store_true', help='debug')
    args.add_argument('--gpu', type=int, default=0, help='gpu')
    args.add_argument('--fine_tuned_unet', default=None, help='fine tuned unet')
    args.add_argument('--concepts_to_remove', default=None, help='List of concepts to remove')
    args.add_argument('--dataset_type', default='concept_removal', help='dataset path')
    args.add_argument('--root-template', default='results/results_seed_%s/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/modularity/art/%s', help='root template')
    args.add_argument('--timesteps', default=51, type=int, help='Timesteps')
    args.add_argument('--batch_size', default=4, type=int, help='Batch size')
    args.add_argument('--style_classifer_path', default='../Diffusion-MU-Attack/results/checkpoint-2800/')

    args = args.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda:%s' % args.gpu)
    # get clip model
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = clip_model.to(device)

    style_classifier = pipeline('image-classification', model=args.style_classifer_path, device=args.gpu)
    print(style_classifier)
    root_dir = os.path.join('benchmarking results', args.fine_tuned_unet, args.dataset_type)

    concept_folders = os.listdir(root_dir)

    results = {}
    sd_feats = {}
    removed_feats = {}

    for folder in concept_folders:
        results[folder] = {}
        results[folder]['sim'] = []
        results[folder]['acc'] = []
        sd_feats[folder] = []
        removed_feats[folder] = []
        path = os.path.join(root_dir, folder)
        # list files that start with sd and end with .png
        files = [f for f in os.listdir(path) if f.startswith('sd') and f.endswith('.png')]
        num_invalid = 0
        for f in files:
            # process image
            image = Image.open(os.path.join(path, f))
            prev_img = clip_processor(images=image, return_tensors='pt')  
            # put keys in prev_img to cuda
            prev_img = {k: v.to(device) for k, v in prev_img.items()}   
            prev_feats = clip_model.get_image_features(**prev_img)
            # normalise
            prev_feats = prev_feats / prev_feats.norm(dim=-1, keepdim=True) 
            sd_feats[folder].append(prev_feats.detach().cpu())

            style_pred_pre = style_classifier(image, top_k=129)[:10]
            prev_label = get_labels(style_pred_pre, folder)
            
            # read removal image
            name = f.replace('sd', 'removed')
            image = Image.open(os.path.join(path, name))
            removal_img = clip_processor(images=image, return_tensors='pt')
            removal_img = {k: v.to(device) for k, v in removal_img.items()}
            removal_feats = clip_model.get_image_features(**removal_img)
            removal_feats = removal_feats / removal_feats.norm(dim=-1, keepdim=True)
            removed_feats[folder].append(removal_feats.detach().cpu())

            style_label_after = get_labels(style_classifier(image, top_k=129)[:top_k[folder]], folder)
            # print(prev_label, style_label_after)
            # if prev_label == 1 and style_label_after == 0:
            #     acc = 1
            #     results[folder]['acc'].append(acc)
            # if prev_label == 1 and style_label_after == 1:
            #     acc = 0
            results[folder]['acc'].append(style_label_after)
            if prev_label == 0:
                num_invalid += 1
                # style_label_after = 0
            
            # get similarity
            sim = torch.nn.functional.cosine_similarity(prev_feats, removal_feats, dim=-1)

            results[folder]['sim'].append(sim.item())

        # Style classifier results
        acc = np.mean(results[folder]['acc'])
        results[folder]['mean_acc'] = acc
        print('Concept:', folder, 'Acc:', acc, 'Num invalid:', num_invalid)

        # Cosine similarity
        mean = np.mean(results[folder]['sim'])
        std = np.std(results[folder]['sim'])
        results[folder]['mean_sim'] = mean
        print('Concept:', folder, 'Mean:', mean, 'Std:', std)

        # save results
        with open(os.path.join(path, 'results.json'), 'w') as f:
            json.dump(results[folder], f)
            
    # print(results)
    # save mean of all artist styles
    all_acc_mean, all_sim_mean = 0, 0

    for iter, folder in enumerate(concept_folders):
        all_acc_mean += results[folder]['mean_acc']
        all_sim_mean += results[folder]['mean_sim']

    all_acc_mean = all_acc_mean / len(concept_folders)
    all_sim_mean = all_sim_mean /len(concept_folders)

    print("Average accuracy", all_acc_mean)
    print("Average similarity", all_sim_mean)
    # save this 
    results = {}
    results['all_acc_mean'] = all_acc_mean
    results['all_sim_mean'] = all_sim_mean
    with open(os.path.join(root_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()
    