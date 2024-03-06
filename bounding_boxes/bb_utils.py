from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


def bb_model(args):
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=args.gpu, trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer

def get_bounding_box_from_response(repsonse):
    # bounding box response is of the format
    # < ref> the white cat</ref><box> (x1, y1),(x2, y2)</box>
    # get the bounding box
    box = repsonse.split('<box>')[1].split('</box>')[0]
    # get the coordinates
    x1, y1, x2, y2 = box.split(',')
    # remove the brackets
    x1, y1 = x1[1:], y1[:-1]
    x2, y2 = x2[1:], y2[:-1]
    # have to half the coordinates because qwen model weirdly returns bounding boxes for a 1024 x 1024 image
    return (int(x1)//2, int(y1)//2), (int(x2)//2, int(y2)//2)
