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
sys.path.append('sparsity')
from eval_coco import CLIPModelWrapper
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')
    adjective = args.modularity['adjective']

    base_prompts, adj_prompts, _ = get_prompts(args)
    print(base_prompts)

    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device=args.gpu)
    model.eval()

    base_root = args.modularity['img_save_path']
    base_prefix = 'base_{}'
    # images after removal
    adj_root = args.modularity['remove_expert_path'] if not args.modularity['condition']['remove_neurons'] else args.modularity['remove_neuron_path']
    adj_prefix = 'img_{}_adj'

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # get the embeddings for the base images
    all_base_imgs = []
    all_adj_imgs = []
    fid = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    for iter in range(len(base_prompts)):
        # print(os.path.join(base_root, base_prefix.format(iter) + '.jpg'), base_prompts[iter])
        # print(os.path.join(adj_root, adj_prefix.format(iter) + '.jpg'), adj_prompts[iter])
        img_base = Image.open(os.path.join(base_root, base_prefix.format(iter) + '.jpg')) # base imahe
        img_adj = Image.open(os.path.join(adj_root, adj_prefix.format(iter) + '.jpg')) # image after removal

        # Transform images to tensor
        img_base = base_transform(img_base).unsqueeze(0)
        img_adj = base_transform(img_adj).unsqueeze(0)

        fid_score = fid(img_adj, img_base)
        print(iter, f"FID score: {fid_score}")

        # convert from 
        # Multiply by 255 to denormalise the images
        
        # img_base = img_base 
        # img_adj = img_adj

    #     all_base_imgs.append(img_base)
    #     all_adj_imgs.append(img_adj)

    # all_base_imgs = torch.cat(all_base_imgs, dim=0)
    # all_adj_imgs = torch.cat(all_adj_imgs, dim=0)
    # convert to uint8
    # all_base_imgs = all_base_imgs.to(torch.uint8)
    # all_adj_imgs = all_adj_imgs.to(torch.uint8)
    # calculate fid
    
    # fid.update(all_base_imgs, real=True)
    # fid.update(all_adj_imgs, real=False)
    # fid_score = fid.compute()
    # fid_score = fid(all_adj_imgs, all_base_imgs)
    # print(f"FID score: {fid_score}")

    # save the FID score
    # with open(os.path.join(args.modularity['remove_neuron_path'], 'fid_score.txt'), 'w') as f:
    #     f.write(str(fid_score))




        # img_base = preprocess(img_base).unsqueeze(0).to(args.gpu)
        # img_adj = preprocess(img_adj).unsqueeze(0).to(args.gpu)

        
    #     base_text_feat = base_prompts[iter]
    #     adj_text_feat = adj_prompts[iter]

    #     with torch.no_grad():
    #         text_embs = clip.tokenize([base_text_feat, adj_text_feat]).to(args.gpu)

    #         logits_per_image, _ = model(img_base, text_embs)
    #         base_img_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #         logits_per_image, _ = model(img_adj, text_embs)
    #         adj_img_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #         base_img_label = base_img_probs.argmax()
    #         adj_img_label = adj_img_probs.argmax()

    #         # consider correct classification for pair if both images are classified correctly
    #         accuracy = (base_img_label == gt_labels[0].item()) and (adj_img_label == gt_labels[1].item())
    #         accuracy = int(accuracy)
    #         if not accuracy:
    #             wrong_samples.append((base_text_feat, adj_text_feat, base_img_probs, adj_img_probs))
    #         avg_score += accuracy

    # print(f"Accuracy: {avg_score/len(base_prompts)}")

    # print("Wrong samples")
    # print(len(wrong_samples))
    # print(wrong_samples)




if __name__ == '__main__':
    main()



