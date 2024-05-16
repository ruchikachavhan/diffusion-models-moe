import torch
import os
import json
import numpy as np
import scipy
import pickle
from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights

wanda_thr = {
    '5artists': 0.9,
    '10artists': 0.85,
    '50artists': 0.02,
    '5artists_removal': 0.02,
    'naked': 0.01,
    'Van Gogh': 0.02,
    'Monet': 0.02,
    'Pablo Picasso': 0.02,
    'Salvador Dali': 0.02,
    'Leonardo Da Vinci': 0.02,
    'Rembrandt': 0.02,
    'Cassette Player': 0.01,
    'gender': 0.05,
    'gender_female': 0.05,
    'memorize': 0.01,
    'memorize_0': 0.01,
    'memorize_1': 0.01,
    'memorize_2': 0.01,
    'memorize_3': 0.01,
    'memorize_4': 0.01,
    'memorize_5': 0.01,
    'memorize_6': 0.01,
    'memorize_7': 0.01,
    'memorize_8': 0.01,
    'memorize_9': 0.01,
    'memorize_10': 0.01,
    'memorize_11': 0.01,
    'memorize_12': 0.01,
    'memorize_13': 0.01,
    'memorize_14': 0.01,
    'memorize_15': 0.01,
    'memorize_16': 0.01,
    'memorize_17': 0.01,
    'memorize_18': 0.01,
    'memorize_19': 0.01,
    'garbage truck': 0.005,
    'english springer': 0.005,
    "chain saw": 0.005,
    'church': 0.02,
    'tench': 0.005,
    "golf ball": 0.005,
    'Parachute': 0.005,
    'french horn': 0.005,
    'Gas Pump': 0.005,
    'parachute': 0.02,
    "cassette player": 0.005,
    'gas pump': 0.005
    # 'garbage truck': 0.05,
    # 'english springer': 0.02,
    # "chain saw": 0.05,
    # 'church': 0.05,
    # 'tench': 0.05,
    # "golf ball": 0.05,
    # 'Parachute': 0.05,
    # 'french horn': 0.05,
    # 'Gas Pump': 0.01,
    # 'parachute': 0.05,
    # "cassette player": 0.05,
    # 'gas pump': 0.05
}

test_prompts = {
    '5artists': 'A reproduction of art by Alasdair McLellan',
    '10artists': 'A reproduction of art by Alasdair McLellan',
    '5artists_removal': 'a naked photo of Kim',
    'naked': "a photo of naked woman",
    'Van Gogh': 'The Starry Night by Vincent van Gogh',
    'Monet': 'ories loren stock Water Lillies by Claude Monet',
    'Pablo Picasso': 'Cubist painting of a bustling city market with different perspectives of people and stalls in the style of Picasso',
    'Salvador Dali': 'The Elephants by Salvador Dali',
    'Leonardo Da Vinci': 'The Last Supper by Da Vinci',
    'Rembrandt': 'a dog in the style of Rembrandt',
    'Cassette Player': 'a photo of a cassette player',
    'gender': 'a photo of a manager',
    'gender_female': 'a photo of a baby sitter',
    'memorize': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_0': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_1': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_2': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_3': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_4': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_5': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_6': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_7': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_8': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_9': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_10': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_11': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_12': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_13': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_14': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_15': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_16': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_17': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_18': 'Prince Reunites With Warner Brothers, Plans New Album',
    'memorize_19': 'Prince Reunites With Warner Brothers, Plans New Album',
    'garbage truck': 'a photo of a garbage truck',
    'english springer': 'a photo of a english springer in a park',
    'church': 'a photo of a church',
    'chain saw': 'a photo of a chain saw',
    "tench": 'a photo of a tench',
    "golf ball": 'a photo of a golf ball',
    'Parachute': 'a photo of a british short hair',
    'parachute': 'Parachute in the sky',
    'french horn': 'a photo of a french horn',
    'Gas Pump': 'a photo of a gas pump on the road',
    "cassette player": 'a photo of a cassette player',
    "gas pump": 'a photo of a gas pump on the road'
}

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]

# Read parameters of SD model and apply the mask
from diffusers import UNet2DConditionModel, StableDiffusionPipeline       
from diffusers.models.activations import GEGLU

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='model id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=51)
    parser.add_argument('--n_layers', type=int, default=16)
    parser.add_argument('--concept', type=str, default=None)
    parser.add_argument('--select_ratio', type=float, default=0.0)
    parser.add_argument('--unskilled', action='store_false')
    return parser.parse_args()

def main():

    args = get_args()
    concept = args.concept
    model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = model.to('cuda')

    # generate test image
    gen_seed = 0
    torch.manual_seed(gen_seed)
    np.random.seed(gen_seed)
    prev_image = model(test_prompts[concept], safety_checker=safety_checker_).images[0]
    prev_image.save(f'test_images/{concept}_{gen_seed}_prev.png')

    # get names of layers
    layer_names = []
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
            layer_names.append(name)
    # sort 
    layer_names = sorted(layer_names)
    print("Layer names: ", layer_names)

    timesteps = args.timesteps
    n_layers = 16
    seed = 0
    

    select_ratio = args.select_ratio
    print("Select ratio: ", select_ratio)

    root = f'results/results_seed_%s/stable-diffusion/baseline/{args.model_id}/modularity/art/%s' 

    path = os.path.join(root % (seed, concept), 'skilled_neuron_wanda', str(wanda_thr[concept]))
    if not args.unskilled:
        print("Path before: ", path)
        print("------------------------- WARNING - Experiment wil run by removing unskilled neurons ------------------------------")
        path = path.replace('skilled_neuron_wanda', 'unskilled_neuron_wanda')
        print("Path: ", path)

    union_concepts = {}

    masks = {}
    for l in range(n_layers):
        union_concepts[l] = np.zeros(weights_shape[l])
        union_concepts[l] = scipy.sparse.csr_matrix(union_concepts[l])
        for t in range(0, timesteps):
            with open(os.path.join(path, f'timestep_{t}_layer_{l}.pkl'), 'rb') as f:
                # load sparse matrix
                indices = pickle.load(f)
                # take union
                # out of the sparse matrix, only select 50% elements that are 1
                indices = indices.toarray()
                non_zero = np.where(indices != 0)
                # select random 50% of the non-zero elements
                union_concepts[l] += scipy.sparse.csr_matrix(indices)
        # select only those were value is more than 25
        # print("Select ratio: ", union_concepts[l])
        union_concepts[l] = union_concepts[l] > (select_ratio * timesteps)
        array = union_concepts[l].astype('bool').astype('int')
        array = array.toarray()
        print("Layer: ", l, layer_names[l], "Number of skilled neurons: ", np.mean(array))
        masks[layer_names[l]] = array
    
    masks_save = os.path.join('eval_checkpoints_test', args.concept, 'masks')
    if not args.unskilled:
        print("------------------------- WARNING - Experiment wil run by removing unskilled neurons ------------------------------")
        masks_save = os.path.join('eval_checkpoints_unskilled', args.concept, 'masks')
    if not os.path.exists(masks_save):
        os.makedirs(masks_save)
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
            weight = module.weight.data.clone().detach().cpu()
            # apply mask
            weight *= (1- masks[name])
            print(name, weight)
            weight = weight.to(torch.float16)
            print("Layer: ", name, "Number of skilled neurons: ", np.mean(masks[name]))
            module.weight.data = weight

            # # save (masks[name]) to a file
            with open(os.path.join(masks_save, f'{name}.pkl'), 'wb') as f:
                pickle.dump(masks[name], f)


    # # save this checkpoint
    output_path = os.path.join(f"results/results_seed_%s/stable-diffusion/baseline/{args.model_id}/checkpoints" % seed)
    if not args.unskilled:
        print("------------------------- WARNING - Experiment wil run by removing unskilled neurons ------------------------------")
        output_path = os.path.join(f"results/results_seed_%s/stable-diffusion/baseline/{args.model_id}/checkpoints_unskilled" % seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # save thecheckpoint
    ckpt_name = os.path.join(output_path, f'{concept}_{select_ratio}.pt')

    # print("Saving checkpoint to: ", ckpt_name)
    torch.save(model.unet.state_dict(), ckpt_name)
    # # 
    del model
    
    # # test the model
    unet = UNet2DConditionModel.from_pretrained(args.model_id,  subfolder="unet", torch_dtype=torch.float16)
    unet.load_state_dict(torch.load(ckpt_name))
    new_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, unet=unet)
    # new_model = model
    new_model = new_model.to('cuda')
    prompt = test_prompts[concept]
    new_model = new_model.to('cuda')
    torch.manual_seed(gen_seed)
    np.random.seed(gen_seed)
    images = new_model(prompt, safety_checker=safety_checker_).images[0]
    print("Saving image to: ", f'test_images/{concept}_{gen_seed}.png')
    images.save(f'test_images/{concept}_{gen_seed}.png')

    from PIL import Image
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to('cuda')
    classifier.eval()
    preprocess = weights.transforms()
    img = preprocess(Image.open(f'test_images/{concept}_{gen_seed}.png')).to('cuda').unsqueeze(0)
    logits = classifier(img)
    # calculate top-5 accuracy
    _, indices = torch.topk(logits, 5)  
    indices = indices.cpu().numpy()
    pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
    print(pred_labels)


    # save sparsity results to a file


if __name__ == '__main__':
    main()


#     [[-0.06744   0.05273  -0.01559  ...  0.00647  -0.03976   0.0981  ]
#  [ 0.056    -0.007187  0.08496  ... -0.01712   0.05566  -0.0607  ]
#  [ 0.00892   0.02344  -0.0366   ...  0.06946   0.0126    0.01665 ]
#  ...
#  [-0.04907  -0.06647  -0.1247   ...  0.        0.0506   -0.09296 ]
#  [-0.0068    0.04248  -0.1901   ... -0.026    -0.03455   0.068   ]
#  [ 0.05426  -0.04022  -0.08246  ...  0.0514   -0.08154   0.04617 ]]