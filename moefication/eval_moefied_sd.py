import json
import os
import sys
import torch
import tqdm
import types
import argparse
import numpy as np
from ast import arg
from PIL import Image
from re import template
from torchvision import transforms
import moe_utils
sys.path.append(os.getcwd())
import utils as dm_utils
import eval_coco as ec
from diffusers.models.activations import GEGLU

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='../COCO-vqa', help='path to the coco dataset')
    parser.add_argument('--blocks-to-change', nargs='+', default=['down_block', 'mid_block', 'up_block'], help='blocks to change the activation function')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--res-path', type=str, default='results/stable-diffusion/', help='path to store the results of moefication')
    parser.add_argument('--dbg', action='store_true', help='debug mode')
    parser.add_argument('--num-images', type=int, default=1000, help='number of images to test')
    parser.add_argument('--fine-tuned-unet', type = str, default = None, help = "path to fine-tuned unet model")
    parser.add_argument('--model-id', type=str, default="runwayml/stable-diffusion-v1-5", help='model id')
    parser.add_argument('--timesteps', type=int, default=51, help='number of denoising time steps')
    parser.add_argument('--num-layer', type=int, default=3, help='number of layers')
    parser.add_argument('--topk-experts', type=float, default=1, help='ratio of experts to select')
    parser.add_argument('--templates', type=str, 
                        default='{}.{}.attentions.{}.transformer_blocks.0.ff.net.0.proj.weight',
                        help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

    args = parser.parse_args()
    return args

class MOENeuronReceiver:
    def __init__(self):
        self.gates = []
        self.hidden_states = []
    
    def hook_fn(self, module, input, output):
        args = (1.0,)
        hidden_states, gate = module.proj(input[0], *args).chunk(2, dim=-1)
        gate = module.gelu(gate)

        if module.patterns is not None:
            k = module.k
            bsz, seq_len, hidden_size = gate.shape
            gate_gelu = gate.clone()
            gate_gelu = gate_gelu.view(-1, hidden_size)
            score = torch.matmul(gate_gelu, module.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            cur_mask = torch.nn.functional.embedding(labels, module.patterns).sum(-2)
            gate[cur_mask == False] = 0
                
        self.gates.append(gate.detach().cpu())
        hidden_states = hidden_states * gate
        return hidden_states
    
    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()
    
    def observe_activation(self, model, img, ann):
        hooks = []
        # reset the gates
        self.gates = []

        # hook the model
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        print(f"Number of modules: {num_modules}")
        # forward pass
        #  fix seed to get the same output for every run
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        return out, self.gates
    
    def test(self, model, ann = 'A brown dog in the snow', relu_condition = False):
        # hook the model
        torch.manual_seed(0)
        np.random.seed(0)
        nochange_out = model(ann).images[0]
        nochange_out.save('nochange_out.png')
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)

        print(f"Number of modules: {num_modules}")
        # forward pass
        #  fix seed to get the same output
        torch.manual_seed(0)
        np.random.seed(0)
        out = model(ann).images[0]

        # remove the hook
        self.remove_hooks(hooks)
        
        # test if all gates have positive values if relu_condition is True
        for gate in self.gates:
            assert torch.all(gate >= 0) == relu_condition, "All gates should be positive"

        # save test image
        out.save('test_image_moe.png')
        self.gates = []



def modify_ffn(ffn, path, k):
    assert type(ffn) == GEGLU
    labels = torch.load(path)
    cluster_num = max(labels)+1
    patterns = []
    for i in range(cluster_num):
        patterns.append(np.array(labels) == i)
    if ffn.proj.weight.dtype == torch.float32:
        ffn.patterns = torch.Tensor(patterns).cuda()
    elif ffn.proj.weight.dtype == torch.float16:
        ffn.patterns = torch.Tensor(patterns).cuda().to(torch.float16)
    # ffn.k is the ratio of selected experts
    ffn.k = int(cluster_num * k)
    print("Moefied model with ", ffn.k, "experts in layer", path)
        
    
def main():
    args = get_args()
    dm_utils.make_dirs(args)
    # make directories to store the results of moefication
    if not os.path.exists(os.path.join(args.res_path, args.model_id, f'moe_images_{args.topk_experts}')):
        os.makedirs(os.path.join(args.res_path, args.model_id, f'moe_images_{args.topk_experts}'))

    model, num_geglu = dm_utils.get_sd_model(args)
    model = model.to(args.gpu)
    
    # Modify FFN to add expert labels
    for name, module in model.unet.named_modules():
        if 'ff.net' in name and isinstance(module, GEGLU):
            ffn_name = name + '.proj.weight'
            path = os.path.join(args.res_path, 'param_split', ffn_name)
            modify_ffn(module, path, args.topk_experts)

    imgs, anns = dm_utils.coco_dataset(args.data_path, 'val', args.num_images)
    neuron_receiver = MOENeuronReceiver()
    neuron_receiver.test(model, relu_condition=args.fine_tuned_unet is not None)
    
    orig_imgs, non_moe_imgs, moe_imgs = [], [], []
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    iter = 0
    for img, ann in tqdm.tqdm(zip(imgs, anns)):
        if iter > 10 and args.dbg:
            break
        print("text: ", ann)
        # fix seed
        torch.manual_seed(0)
        np.random.seed(0)
        # Without MOEfication
        out = model(ann).images[0]
        # With MOEfication
        out_moe, gates_moe = neuron_receiver.observe_activation(model, img, ann)

        if iter < 10:
            out.save(os.path.join(args.res_path, args.model_id, f'moe_images_{args.topk_experts}', f'original_{iter}.png'))
            out_moe.save(os.path.join(args.res_path, args.model_id, f'moe_images_{args.topk_experts}', f'moe_{iter}.png'))

        orig_imgs.append(transform(Image.open(img).convert('RGB')))
        non_moe_imgs.append(transform(out.convert('RGB')))
        moe_imgs.append(transform(out_moe.convert('RGB')))
        iter += 1

    orig_imgs = torch.stack(orig_imgs) * 255
    non_moe_imgs = torch.stack(non_moe_imgs) * 255
    moe_imgs = torch.stack(moe_imgs) * 255
    fid = ec.calculate_fid(orig_imgs, non_moe_imgs)
    fid_moe = ec.calculate_fid(orig_imgs, moe_imgs)
    print(f"FID: {fid}, FID MOE: {fid_moe}")
    # save the fid scores
    with open(os.path.join(args.res_path, args.model_id, f'fid_{args.topk_experts}.txt'), 'w') as f:
        f.write(f"FID: {fid}, FID MOE: {fid_moe}")


if __name__ == "__main__":
    main()
