import os
import sys
import json
import tqdm
import torch
import numpy as np
from PIL import Image
import clip
from mod_utils import get_prompts
sys.path.append(os.getcwd())
import utils

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')

    clip_oracle = False

    adjective = args.modularity['adjective']

    base_prompts, adj_prompts, _ = get_prompts(args)
    print(base_prompts)

    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device=args.gpu)
    model.eval()

    if clip_oracle:
        save_path = args.modularity['img_save_path']
        gt_labels = torch.tensor([0, 1]).to(args.gpu)
        prefix = ['base_{}', 'adj_{}']
    else:
        save_path=args.modularity['remove_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path']
        gt_labels = torch.tensor([1, 0]).to(args.gpu)
        prefix = ['img_{}', 'img_{}_adj']


    # load path to images 
    images = os.listdir(save_path)

    # separate base and adj images
    base_images = sorted([img for img in images if 'adj' not in img])
    adj_images = sorted([img for img in images if 'adj' in img])
    wrong_samples =[]

    avg_score = 0

    # get the embeddings for the base images
    for iter in range(len(base_prompts)):
        print(os.path.join(save_path, prefix[0].format(iter) + '.jpg'), base_prompts[iter])
        img_base = Image.open(os.path.join(save_path, prefix[0].format(iter) + '.jpg'))
        img_adj = Image.open(os.path.join(save_path, prefix[1].format(iter) + '.jpg'))


        img_base = preprocess(img_base).unsqueeze(0).to(args.gpu)
        img_adj = preprocess(img_adj).unsqueeze(0).to(args.gpu)
        base_text_feat = base_prompts[iter]
        adj_text_feat = adj_prompts[iter]

        with torch.no_grad():
            text_embs = clip.tokenize([base_text_feat, adj_text_feat]).to(args.gpu)

            logits_per_image, _ = model(img_base, text_embs)
            base_img_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            logits_per_image, _ = model(img_adj, text_embs)
            adj_img_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            base_img_label = base_img_probs.argmax()
            adj_img_label = adj_img_probs.argmax()

            # consider correct classification for pair if both images are classified correctly
            accuracy = (base_img_label == gt_labels[0].item()) and (adj_img_label == gt_labels[1].item())
            accuracy = int(accuracy)
            if not accuracy:
                wrong_samples.append((base_text_feat, adj_text_feat, base_img_probs, adj_img_probs))
            avg_score += accuracy

    print(f"Accuracy: {avg_score/len(base_prompts)}")

    print("Wrong samples")
    print(len(wrong_samples))
    print(wrong_samples)




if __name__ == '__main__':
    main()



