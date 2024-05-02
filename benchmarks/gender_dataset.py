import torch
import numpy as np
import os
import json
from PIL import Image, ImageFilter
import sys
sys.path.append(os.getcwd())
import utils
from torchmetrics.multimodal import CLIPImageQualityAssessment as CLIPIQA
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import pandas as pd
from torchvision import transforms

# Read a csv file and generate a list of professions
# RUn SD model on the list of professions
# Use CLIp to classify if it is male or female
# save the results to a file

def main():
    model_id = 'runwayml/stable-diffusion-v1-5'
    seed = 0
    gpu = 7

    # Load the model
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(gpu)

    metric = CLIPIQA(model_name_or_path='clip_iqa', prompts=("quality", ("photo of a woman", "photo of a man")))

    file_path = 'modularity/datasets/profession200_prompts.csv'

    # Read the csv file
    professions = pd.read_csv(file_path)
    professions = professions['profession'].values.tolist()
    professions = np.unique(professions)

    print(f'Professions: {professions}')

    base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    male_professions, female_professions = [], []
    for iter, profession in enumerate(professions):
        prompt = f'photo of a {profession}'
        print(f'Prompt: {prompt}')

        image = model(prompt).images[0]
        image = base_transform(image).unsqueeze(0) 
        # put image thourgh CLIP
        clip_score = metric(image)
        print(f'Clip score: {clip_score}')

        if clip_score['user_defined_0'] > 0.5:
            female_professions.append(profession)
        else:
            male_professions.append(profession)

    # save the results
    with open('modularity/datasets/male_professions.txt', 'w') as f:
        for profession in male_professions:
            f.write(f'{profession}\n')
    
    with open('modularity/datasets/female_professions.txt', 'w') as f:
        for profession in female_professions:
            f.write(f'{profession}\n')

if __name__ == '__main__':
    main()

