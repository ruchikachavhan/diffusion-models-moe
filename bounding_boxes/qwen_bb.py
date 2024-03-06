from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import cv2
import numpy as np
import json
torch.manual_seed(1234)
from bb_utils import bb_model, get_bounding_box_from_response
from save_bb import get_bounding_box, get_bounding_box_latent, save_bounding_box
# Note: The default behavior now has injection attack prevention off.



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

image_path = 'test_images/nochange_out.png'
# read the image 
img = cv2.imread(image_path)
print(img.shape)
query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": "what is this?"},
    ])

# Generate response
response, history = model.chat(tokenizer, query=query, history=None)

print("Response:", response)

# Get bounding box coordinates
response, history = model.chat(tokenizer, 'Frame the location of the brown dog', history=history)

print("Response:", response)

top_left, bottom_right = get_bounding_box_from_response(response)
print(top_left, bottom_right)
# draw points on img
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
cv2.imwrite('test_images/bbox_cv2.png', img)

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('test_images/bbox.png')
else:
  print("no box")

layer_names = ['down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.weight', 
              'up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.weight']

latent_space_seq_length = [4096, 4096, 1024, 1024, 256, 256, 64, 256, 256, 256, 1024, 1024, 1024, 4096, 4096, 4096]
latent_space_seq_length = [np.sqrt(x) for x in latent_space_seq_length]
default_img_size = 512

grids = get_bounding_box_latent(layer_names, top_left, bottom_right, latent_space_seq_length, default_img_size)
# save the bounding box
#  save the coordinates
with open('test_images/test_bbox.json', 'w') as f:
  json.dump(grids, f)