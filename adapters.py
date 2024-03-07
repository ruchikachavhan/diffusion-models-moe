import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False


def set_module_grad_status(module, flag=False):
    if isinstance(module, list):
        print("list", module)
        for m in module:
            set_module_grad_status(m, flag)
    else:
        print("not a list", module)
        for p in module.parameters():
            p.requires_grad = flag

def check_tunable_params(model, verbose=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if verbose:
                print(name)
            trainable_params += param.numel()

    # return  f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f} + \n"
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f}"
    )

    return 100 * trainable_params / all_param

## Attention Tuning
def enable_attention_update(model):
    print("Enabling Attention layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if "attentions" in name:
                param.requires_grad = True

## Norm Tuning
def enable_norm_update(model):
    print("Enabling Normalization layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if "norm" in name:
                param.requires_grad = True

## Bias Tuning
def enable_bias_update(model):
    print("Enabling Bias layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if name == "bias":
                param.requires_grad = True



def get_adapted_unet(unet, method, args, **kwargs):
    if method == "full":
        verbose = False
        pass
    elif method == "attention":
        disable_grad(unet)
        enable_attention_update(unet)
        verbose = True
    elif method == "norm":
        disable_grad(unet)
        enable_norm_update(unet)
        verbose = True
    elif method == "bias":
        disable_grad(unet)
        enable_bias_update(unet)
        verbose = True
    elif method == "norm_bias":
        disable_grad(unet)
        enable_norm_update(unet)
        enable_bias_update(unet)
        verbose = True
    elif method == "norm_bias_attention":
        disable_grad(unet)
        enable_norm_update(unet)
        enable_bias_update(unet)
        enable_attention_update(unet)
        verbose = True
    else:
        raise ValueError("Fine-Tuning Method not defined")

    return unet