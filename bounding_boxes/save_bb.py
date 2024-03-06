import json
import os
import sys
import torch
import cv2
import numpy as np
sys.path.append(os.getcwd())
import utils
from neuron_receivers import SparsityMeasure
sys.path.append('moefication')
from helper import modify_ffn_to_experts
sys.path.append('bounding_boxes')
from bb_utils import bb_model, get_bounding_box_from_response


def get_bounding_box(img, bb_model, tokenizer, prompt, question=f"Frame the location of the "):
    # Save image temp, TODO: Find a better solution
    print(img)
    img.save('temp.jpg')
    # read temp.jpg with numpy
    img_test = cv2.imread('temp.jpg')
    print(img_test.shape)
    print("text: ", question + ' '.join(prompt.split(' ')[1:]))
    # get the bounding box
    query = tokenizer.from_list_format([
        {"image": 'temp.jpg'},
        {"text": question + ' '.join(prompt.split(' ')[1:])},
    ])

    # Get bounding box coordinates
    response, history = bb_model.chat(tokenizer, query, history=None)

    print("Response:", response)

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        image.save('test_images/bbox.png')
    else:
        print("no box")

    top_left, bottom_right = get_bounding_box_from_response(response)
    return top_left, bottom_right
    
def get_bounding_box_latent(layer_names, top_left_coord, bottom_right_coord, latent_space_seq_length, default_img_size=512):
    
    # The latent space of the model is 4, 64, 64 in the first layer
    # Output of the SD model is 3, 512, 512
    # So every 1 x 1 x 4 block of the latent space corresponds to 8 x 8 x 3 block of the output space
    # So, to get corresponding latent space token, we divide the top_left and bottom_right by 8
    grids = {}
    for i, width in enumerate(latent_space_seq_length):
        factor = default_img_size // width
        top_left = (top_left_coord[0] // factor, top_left_coord[1] // factor)
        bottom_right = (bottom_right_coord[0] // factor, bottom_right_coord[1] // factor)
        # exchange the x and y coordinates to match the latent space
        # in the image x coordinate cooresponds to the width and y coordinate corresponds to the height
        # but in tensor x coordinate corresponds to the height and y coordinate corresponds to the width
        top_left = (top_left[1], top_left[0])
        bottom_right = (bottom_right[1], bottom_right[0])

        # make a grid in latent space coordinates with top_left and bottom_right
        x = torch.arange(top_left[0], bottom_right[0])
        y = torch.arange(top_left[1], bottom_right[1])
        xx, yy = torch.meshgrid(x, y)
        # stack the x and y coordinates to get the grid
        grid = torch.stack((xx, yy), 2)
        grid = grid.view(-1, 2)
        # convery every coordinate of latent space to corresponding point in flattened latent space
        grid = grid[:, 0] * latent_space_seq_length[i] + grid[:, 1] 
        grids[layer_names[i]] = grid.tolist()
    return grids

def save_bounding_box(model, BBModel, tokenizer, propmts, args, neuron_receiver, latent_space_seq_length, layer_names, prefix, default_img_size=512):
    bb_coordinates_layer = {}
    for ann in propmts:
        print("text: ", ann)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0) 
        # run model for the original text
        out, _ = neuron_receiver.observe_activation(model, ann)

        # get bounding box
        top_left_coord, bottom_right_coord = get_bounding_box(out, BBModel, tokenizer, ann)
        # get the bounding box in latent space
        grids = get_bounding_box_latent(layer_names, top_left_coord, bottom_right_coord, latent_space_seq_length) 
        bb_coordinates_layer[ann] = grids
        
    print("Saving results")
    print(bb_coordinates_layer)
    
    # save the coordinates
    with open(os.path.join(args.save_path, f'bb_coordinates_layer_{prefix}.json'), 'w') as f:
        json.dump(bb_coordinates_layer, f)

def main():
    args = utils.Config('experiments/mod_config.yaml', 'modularity')
    args.configure('modularity')

    # model 
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)

    model, layer_names, num_experts_per_ffn = modify_ffn_to_experts(model, args)
    print(layer_names)

    # Boundin box model
    BBModel, tokenizer = bb_model(args)

    neuron_receiver = SparsityMeasure()

    # Dataset from things.txt
    # read things.txt
    with open('modularity/things.txt', 'r') as f:
        objects = f.readlines()
    base_prompts = [f'a {thing.strip()}' for thing in objects]
    adjectives = args.modularity['adjective']
    adj_prompts = [f'a {adjectives} {thing}' for thing in objects]
    
    # In the Unet, downsampling and upsampling changes the sequence length of the query
    # latent_space_seq_length is the sequence length of the latent space at different layers
    latent_space_seq_length = [4096, 4096, 1024, 1024, 256, 256, 64, 256, 256, 256, 1024, 1024, 1024, 4096, 4096, 4096]
    latent_space_seq_length = [np.sqrt(x) for x in latent_space_seq_length]
    default_img_size = 512

    # save bounding box coordinates
    save_bounding_box(model, BBModel, tokenizer, adj_prompts, args, neuron_receiver, latent_space_seq_length, layer_names, 'adj')

    # save bounding box coordinates
    save_bounding_box(model, BBModel, tokenizer, base_prompts, args, neuron_receiver, latent_space_seq_length, layer_names, 'base')
   


if __name__ == '__main__':
    main()