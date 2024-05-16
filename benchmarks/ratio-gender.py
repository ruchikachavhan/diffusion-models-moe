import torch
import numpy as np
import os
from PIL import Image
import json
from transformers import CLIPProcessor, CLIPModel

texts = ['a photo of a man', 'a photo of a woman']

def get_clip_output(model, processor, image, texts):
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    # put inputs to gpu
    inputs = {name: tensor.to('cuda:5') for name, tensor in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits_per_image
    pred_label = logits.softmax(dim=-1).argmax()
    return texts[pred_label.item()]

def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.to('cuda:5')

    f_name = 'modularity/datasets/test_professions.txt'
    # read the file
    professions = []
    with open(f_name, 'r') as f:
        for line in f:
            professions.append(line.strip())
    
    print("Professions", professions)
    root_path = 'benchmarking results/union-timesteps/test_professions'

    results, original_ratio = {}, {}
    for p in professions:
        results[p] = {}
        original_ratio[p] = {}
        results[p]['a photo of a man'] = 0
        results[p]['a photo of a woman'] = 0

        original_ratio[p]['a photo of a man'] = 0
        original_ratio[p]['a photo of a woman'] = 0

    seeds = np.arange(0, 100, 1)
    
    deltas = {}

    all_results = []
    for prof in professions:
        results[prof] = {}
        print("Prof", prof)
        success_rate_m2f= 0
        success_rate_f2m = 0
        num_males_orig = 0
        num_females_orig = 0
        num_female_male_applied = 0
        num_male_female_applied = 0
        success_rate_preserve_female_m2f = 0
        success_rate_preserve_male_f2m = 0
        
        for seed in seeds:
            orig_im1 = os.path.join(root_path, 'gender', f'seed_{seed}', f'gender_{prof}_original.jpg')
            orig_im2 = os.path.join(root_path, 'gender_female', f'seed_{seed}', f'gender_female_{prof}_original.jpg')

            gen_im1 = os.path.join(root_path, 'gender', f'seed_{seed}', f'gender_{prof}.png')
            gen_im2 = os.path.join(root_path, 'gender_female', f'seed_{seed}', f'gender_female_{prof}.png')

            # process the image
            pred_label_orig1 = get_clip_output(model, processor, Image.open(orig_im1), texts)
            pred_label_gen1 = get_clip_output(model, processor, Image.open(gen_im1), texts)

            pred_label_orig2 = get_clip_output(model, processor, Image.open(orig_im2), texts)
            pred_label_gen2 = get_clip_output(model, processor, Image.open(gen_im2), texts)

            original_ratio[prof][pred_label_orig1] += 1
            original_ratio[prof][pred_label_orig2] += 1

            # if original gender is man and applying female generating filter changes gender, then m2f filter is successfull
            if pred_label_orig1 != pred_label_gen1 and pred_label_orig1 == 'a photo of a man':
                success_rate_m2f += 1 
                num_males_orig += 1
            # if original label is man and applying female generating filter doesn't change gender, then m2f fileter is unsuccesful
            if pred_label_orig1 == pred_label_gen1 and pred_label_orig1 == 'a photo of a man':
                num_males_orig += 1

            # if original gender is female and the female generating filter does not changes the gender, then the m2f milter will generate a female
            if pred_label_orig1 == pred_label_gen1 and pred_label_orig1 == 'a photo of a woman':
                success_rate_preserve_female_m2f += 1
                num_female_male_applied += 1
            if pred_label_orig2 != pred_label_gen2 and pred_label_orig2 == 'a photo of a man':
                num_female_male_applied += 1
         
                
            # if original gender is female and applying male generating filter changes gender, then f2m filter is successfull
            if pred_label_orig2 != pred_label_gen2 and pred_label_orig2 == 'a photo of a woman':
                success_rate_f2m +=1 
                num_females_orig += 1
            # if original label is female and applying male generating filter doesn't change gender, then f2m fileter is unsuccesful
            if pred_label_orig2 == pred_label_gen2 and pred_label_orig2 == 'a photo of a woman':
                num_females_orig += 1

            # if original gender is male and the male generating filter does not changes the gender, then the f2m milter will generate a male
            if pred_label_orig2 == pred_label_gen2 and pred_label_orig2 == 'a photo of a man':
                success_rate_preserve_male_f2m += 1
                num_male_female_applied += 1
            if pred_label_orig2 != pred_label_gen2 and pred_label_orig2 == 'a photo of a man':
                num_male_female_applied += 1
         
        # calculate gender ratio of original stable diffusion
                
        p_m = original_ratio[prof]['a photo of a man']
        p_f = original_ratio[prof]['a photo of a woman']

        # calculate success_rate of male to female filter
        success_rate_m2f /= num_males_orig if num_males_orig != 0 else 1.0
        success_rate_f2m /= num_females_orig if num_females_orig!= 0 else 1.0

        success_rate_preserve_female_m2f /= num_female_male_applied if num_female_male_applied != 0 else 1.0
        success_rate_preserve_male_f2m /= num_male_female_applied if num_male_female_applied !=0 else 1.0

        print("original gender ratio", p_m, p_f)
        print("SUccess rates", success_rate_m2f, success_rate_f2m)
        print("SUccess rates preservation", success_rate_preserve_female_m2f, success_rate_preserve_male_f2m)

        # save success rates for every profession in a file

        results[prof]['success_rate_m2f'] = success_rate_m2f
        results[prof]['success_rate_f2m'] = success_rate_f2m

        results[prof]['success_rate_preserve_female_m2f'] = success_rate_preserve_female_m2f
        results[prof]['success_rate_preserve_male_f2m'] = success_rate_preserve_male_f2m


        print(results[prof])

        all_results.append(results[prof])

    # save all results in json file
    
    with open('benchmarking results/union-timesteps/test_professions/results.json', 'w') as f:
        json.dump(all_results, f)
 

        
        # p_new_m = p_f * 0.5 * success_rate_f2m + p_m * 0.5 + p_m * 0.5 + p_m * 0.5 * success_rate_preserve_male_f2m + p_m * 0.5 * (1 - success_rate_m2f)
        # p_new_f = p_m * 0.5 * success_rate_m2f + p_f * 0.5 + p_f * 0.5 + p_f * 0.5 * success_rate_preserve_female_m2f + p_f * 0.5 * (1 - success_rate_f2m)
        
        # p_new_m = p_new_m
        # p_new_f = p_new_f
        # new_gender_ratio = p_new_f/p_new_m if p_new_m > p_new_f else p_new_m/p_new_f
        # print(p_new_m, p_new_f, new_gender_ratio)

    #     success_rate = success_rate/num_valid
    #     print(prof, original_ratio[prof])
    #     print(prof, results[prof])
    #     num_men = original_ratio[prof]['a photo of a man']
    #     num_women = original_ratio[prof]['a photo of a woman']
    #     sd_gender_ratio = num_women/num_men  if num_women < num_men else num_men/num_women
    #     print(prof, sd_gender_ratio, success_rate)
        
    #     p_actual = (1 - success_rate) * sd_gender_ratio + success_rate * 0.5
    #     delta = abs(p_actual - 0.5)/0.5

    #     deltas[prof] = delta    
    # avg_delta = 0.0
    # for k, v in deltas.items():
    #     avg_delta += v
    # avg_delta /= len(deltas)
    # print("Average delta", avg_delta)
    # # save results
   
            

if __name__ == '__main__':
    main()
            

