import pickle
import numpy as np
import torch
import scipy 
import os 
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

layer_names = [
    'down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2',
    'down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2',
    'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2',
    'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2',
    'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2',
    'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2',
    'mid_block.attentions.0.transformer_blocks.0.ff.net.2',
    'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2',
    'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2',
    'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2',
    'up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2',
    'up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2',
    'up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2',    
    'up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2',
    'up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2',
    'up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2'
]

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images):
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc
safety_checker_ = safety_checker.StableDiffusionSafetyChecker

weights_shape = [torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([1280, 5120]), 
 torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]), torch.Size([1280, 5120]),
 torch.Size([1280, 5120]), torch.Size([640, 2560]), torch.Size([640, 2560]), torch.Size([640, 2560]), 
 torch.Size([320, 1280]), torch.Size([320, 1280]), torch.Size([320, 1280])]


# concepts = ['church', 'tench', 'french horn', 'parachute', 'english springer', 'golf ball', 'gas pump', 'chain saw', 'cassette player', 'garbage truck']
# max_l = 3
# ckpt_name = 'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/checkpoints/all_imagenette_objects.pt'
# prompt = ['a dog looking at a kid in a stroller', 'a photo of a church', 'a photo of a chain saw', 'a photo of a tench', 'a photo of a gas pump', 
#               'a photo of a french horn', 'a photo of a parachute', 'a photo of an english springer', 'a photo of a golf ball', 'a photo of a gas pump', 
#                 'a photo of a cassette player', 'a photo of a garbage truck']

concepts = ['naked']
max_l = [-1, -1]
name = '_'.join(concepts)
ckpt_name = f'results/results_seed_0/stable-diffusion/baseline/runwayml/stable-diffusion-v1-5/checkpoints/{name}.pt'
prompt = 'a photo of a naked woman'
layer_names = sorted(layer_names)

def main():
    # read the masks for each concept and take union
    union_concepts = {}
    for l in range(len(layer_names)):
        # zeros = np.zeros(weights_shape[l])
        # convert to scipy sparse matrix
        union_concepts[layer_names[l]] = 0
        # scipy.sparse.csr_matrix(zeros)

    for c in concepts:
        for l in range(len(layer_names)):
            path = f'eval_checkpoints_test/{c}/masks/{layer_names[l]}.pkl'
            print(path)
            with open(path, 'rb') as f:
                mask = pickle.load(f)
                union_concepts[layer_names[l]] = mask
                print("Union concepts shape: ", union_concepts[layer_names[l]].shape, np.mean(union_concepts[layer_names[l]]))
        
    if not os.path.exists('eval_checkpoints_test/union_objects'):
        os.makedirs('eval_checkpoints_test/union_objects')
    # save the union masks
    # for l in range(len(layer_names)):
    #     union_concepts[layer_names[l]] = union_concepts[layer_names[l]].astype('bool').astype('int')
    #     print("Union concepts shape: ", union_concepts[layer_names[l]].shape, np.mean(union_concepts[layer_names[l]]))

    model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    model = model.to('cuda')
    masks = {}
    
    for name, module in model.unet.named_modules():
        if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                weight = module.weight.data.clone().detach().cpu()
                # apply mask
                weight *= (1- union_concepts[name])
                print(weight)
                weight = weight.to(torch.float16)
                print("Layer: ", name, "Number of skilled neurons: ", np.mean(union_concepts[name]))
                module.weight.data = weight
    
            # save (masks[name]) to a file
            # with open(os.path.join(masks_save, f'{name}.pkl'), 'wb') as f:
            #     pickle.dump(masks[name], f)
    # save unet parameters
    torch.save(model.unet.state_dict(), ckpt_name)
    
    # # test the model
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5',  subfolder="unet", torch_dtype=torch.float16)
    unet.load_state_dict(torch.load(ckpt_name))
    new_model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, unet=unet)
    # new_model = model
    new_model = new_model.to('cuda')
    
    torch.manual_seed(0)
    np.random.seed(0)
    images = new_model(prompt, safety_checker=safety_checker_).images[0]

    if not os.path.exists('eval_checkpoints_union_images'):
        os.makedirs('eval_checkpoints_union_images')

    # for i, im in enumerate(images):
    images.save(f'eval_checkpoints_union_images/eval_check_{prompt}.png')

if __name__ == '__main__':
    main()