import pandas as pd
import os
import sys

data1 = pd.read_csv('modularity/datasets/artwork_prompts.csv')
print(data1.keys())
f2 = open('modularity/datasets/i2p_prompts_seed_0_CompVis_stable-diffusion-v1-4.txt', 'r')
data2 = f2.readlines()
length = len(data1)
# change column name from 'artist' to 'concept'
data1.rename(columns={'artist': 'concept'}, inplace=True)
# add prompts from data2 to rows in data1 

for i, prompt in enumerate(data2):
    data1 = data1._append({'case_number': length + i, 'prompt': prompt.split("\n")[0], 'evaluation_seed': 0, 'concept': 'naked'}, ignore_index=True)

data1.to_csv('modularity/datasets/combined_prompts.csv', index=False)

