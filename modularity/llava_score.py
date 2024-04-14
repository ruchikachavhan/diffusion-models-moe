import requests
import os
from PIL import Image
from mod_utils import get_prompts
import torch
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration
sys.path.append(os.getcwd())
import utils

model_id = "llava-hf/llava-1.5-7b-hf"

def llava_prompt(objects, adjective):
    # stitch all objects into a string to form the prompt
    objects_list = ''
    for it in range(len(objects)):
        objects_list += f'({it}) {objects[it]} '
    styles = [f'(1)' + adjective, f'(2) photo']
    style_prompt = ' '.join(styles)
    # randomly shuffle
    random.shuffle(styles)
    # Question to detect object
    prompt = f"USER: <image>\nWhat is the main object in this image? \
        Answer from the following options: {objects_list} . Answer the number \
        only and do not include any other texts (e.g., 1).” \nASSISTANT: <answer1> \
        USER: What is the style of the object? Answer from the fpllowing options: {styles}. \
        Answer the number only and do not include any other texts (e.g., 1).” \nASSISTANT: <answer2>"
    return prompt

def main():
    args = utils.Config('experiments/remove_skills.yaml', 'modularity')
    args.configure('modularity')
    oracle = True
    adjective = args.modularity['adjective']
    # read file name with objects in it
    with open(args.modularity['file_name'], 'r') as f:
        objects = f.readlines()
    objects = [obj.strip() for obj in objects]

    print("Number of objects to test on:", len(objects))

    # LLAVA model
    model = LlavaForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)

    if oracle:
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
    print(base_images)
    wrong_samples =[]

    avg_score = 0

    llava_prompt = llava_prompt(objects, adjective)
    print("Prompt:", llava_prompt)

    # get the embeddings for the base images
    for iter in range(len(objects)):

        # read base image and prompt
        image_file = os.path.join(save_path, prefix[0].format(iter) + '.jpg')
        base_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(llava_prompt, base_image, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Response:", output)
        # img_base = os.path.join(save_path, prefix[0].format(iter) + '.jpg')
        # img_adj = os.path.join(save_path, prefix[1].format(iter) + '.jpg')

        # base_text_feat = 'a photo of ' + base_prompts[iter]
        # object = base_prompts[iter].split(' ')[-1]

        # print("Base:", base_text_feat)
        # query = tokenizer.from_list_format([
        # {"image": img_adj},
        # {"text": "What is this?"},
        #  ])
        
        # with torch.no_grad():
        #     response, history = model.chat(tokenizer, query=query, history=None)
        #     print("Response:", response)
        #     # response, history = model.chat(tokenizer, 'Is the' + object + adjective + "? Answer in yes or no", history=history)
        #     response, history = model.chat(tokenizer, 'Is this a painting of a ' + object + "? Answer in yes or no", history=history)
        #     print("Response:", response)
            
            
if __name__ == '__main__':
    main()
        

