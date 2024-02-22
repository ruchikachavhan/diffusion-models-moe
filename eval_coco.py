import json
import os
import sys
import torch
import tqdm
import torchvision
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from check_sparsity import coco_dataset
from utils import make_dirs, get_sd_model


class CLIPModelWrapper(torch.nn.Module):
    def __init__(self, device):
        super(CLIPModelWrapper, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self.device = device

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        image_output = self.model.get_image_features(**inputs)
        # normalise
        image_output = image_output / image_output.norm(dim=-1, keepdim=True)
        return image_output

def calculate_fid(original_images, generated_images):
    fid = FID(feature=CLIPModelWrapper('cpu'), normalize=False)
    fid.update(original_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()


def evaluate_sd_model(model, imgs, anns, args, transform):
    texts = []
    original_images, generated_images = [], []
    iter = 0
    for img, text in tqdm.tqdm(zip(imgs, anns)):
        if args.dbg and iter > 5:
            break
        # genearate image from prompt
        # fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        generated_image = model(prompt=text).images[0]
        original_image = transform(Image.open(img).convert('RGB'))
        generated_image = transform(generated_image.convert('RGB'))
        original_images.append(original_image)
        generated_images.append(generated_image)
        if iter < 50:
            # Save some images
            torchvision.utils.save_image(original_image, os.path.join(args.res_path, args.model_id, 
                                                        'images', 'evaluation_coco', f'original_{iter}.png'))
            torchvision.utils.save_image(generated_image, os.path.join(args.res_path, args.model_id, 
                                                        'images', 'evaluation_coco', f'generated_{iter}.png'))
            texts.append(text)
        iter += 1

    # calculate FID
    # multiply by 255 to avoid FID scaling warning
    original_images = torch.stack(original_images) * 255
    generated_images = torch.stack(generated_images) * 255
    fid = calculate_fid(original_images, generated_images)
    return fid, texts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../COCO-vqa', help='path to the coco dataset')
    parser.add_argument('--blocks-to-change', nargs='+', default=['down_block', 'mid_block', 'up_block'], help='blocks to change the activation function')
    parser.add_argument('--res-path', type=str, default='results/stable-diffusion/', help='path to store the results of moefication')
    parser.add_argument('--dbg', action='store_true', help='debug mode')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num-images', type=int, default=1000, help='number of images to test')
    parser.add_argument('--fine-tuned-unet', type = str, default = None, help = "path to fine-tuned unet model")
    parser.add_argument('--model-id', type=str, default="runwayml/stable-diffusion-v1-5", help='model id')
    parser.add_argument('--change-activation', action='store_true', help='change the activation function')
    parser.add_argument('--calculate-fid-original', action='store_true', help='calculate FID')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    make_dirs(args)

    imgs, anns = coco_dataset(args.data_path, 'val', args.num_images)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    model, num_geglu = get_sd_model(args) 
    model = model.to(args.gpu)

    if args.calculate_fid_original: 
        assert args.fine_tuned_unet is None, "Pre-trained model needs to be loaded here"
        fid, texts = evaluate_sd_model(model, imgs, anns, args, transform)
        with open(os.path.join(args.res_path, args.model_id, 'images', 'evaluation_coco', 'fid_scores.txt'), 'w') as f:
            f.write(f"FID score: {fid}\n")
        with open(os.path.join(args.res_path, 'texts.txt'), 'w') as f:
            for text in texts:
                f.write(text + '\n')

    if args.change_activation:
        assert args.fine_tuned_unet is not None, "Please provide the path to the fine-tuned unet model"
        fid, _ = evaluate_sd_model(model, imgs, anns, args, transform)
        with open(os.path.join(args.res_path, args.model_id, 'images', 'evaluation_coco', 'fid_scores.txt'), 'w') as f:
            f.write(f"FID score: {fid}\n")
    

if __name__ == '__main__':
    main()
    