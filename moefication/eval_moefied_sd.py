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
from helper import modify_ffn_to_experts
sys.path.append(os.getcwd())
import utils
import eval_coco as ec
from relufy_model import BaseNeuronReceiver
from diffusers.models.activations import GEGLU

class MOEFy(BaseNeuronReceiver):
    def __init__(self):
        super(MOEFy, self).__init__()

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
    
    def test(self, model, ann = 'A brown dog in the snow', relu_condition = False):
        # hook the model
        torch.manual_seed(0)
        np.random.seed(0)
        nochange_out = model(ann).images[0]
        nochange_out.save('test_images/nochange_out.png')
        hooks = []
        num_modules = 0
        for name, module in model.unet.named_modules():
            if isinstance(module, GEGLU) and 'ff.net' in name:
                hook = module.register_forward_hook(self.hook_fn)
                num_modules += 1
                hooks.append(hook)
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
        out.save('test_images/test_image_moe.png')
        self.gates = []
        
    
def main():
    args = utils.Config('experiments/config.yaml', 'moefication')
    args.configure('moefication')

    topk = float(sys.argv[1])
    if topk is not None:
        args.moefication['topk_experts'] = topk
        args.configure('moefication')
    print(f"Topk experts: {args.moefication['topk_experts']}")

    # Model
    model, num_geglu = utils.get_sd_model(args)
    model = model.to(args.gpu)
    
    # Change FFNS to a mixture of experts
    model = modify_ffn_to_experts(model, args)
    
    # Eval dataset
    imgs, anns = utils.coco_dataset(args.dataset['path'], 'val', args.inference['num_images'])

    # MOEFIER
    moefier = MOEFy()
    moefier.test(model, relu_condition=args.fine_tuned_unet is not None)
    
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
        out_moe, _ = moefier.observe_activation(model, ann)

        if iter < 10:
            out.save(os.path.join(args.moefication['img_save_path'], f'original_{iter}.png'))
            out_moe.save(os.path.join(args.moefication['img_save_path'], f'moe_{iter}.png'))

        # Collect images
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
    topk_experts = args.moefication['topk_experts']
    with open(os.path.join(args.save_path, f'fid_{topk_experts}.txt'), 'w') as f:
        f.write(f"FID: {fid}, FID MOE: {fid_moe}")


if __name__ == "__main__":
    main()
