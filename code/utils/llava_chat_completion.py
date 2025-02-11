import argparse
import torch

#llava
import sys
# 导入 log 模块目录
sys.path.append("/LLaVA-main")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image




def Llama_generate(llava_tokenizer,
                   llava_model, 
                   llava_image_processor, 
                   llava_context_len,
                   image_file,
                   messages,
                   temperature: float = 0.2,
                   max_new_tokens: int = 512,
                ):

    model_name='llava-v1.6-mistral-7b-hf_lora_ptb_test'

   
    tokenizer, model, image_processor, context_len = llava_tokenizer,llava_model, llava_image_processor, llava_context_len

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    flag=0
    for message in messages[0]:
        if message['role'] == 'system':
            conv.system = message['content']
        elif message['role'] == "user":
            if flag==1 and model.config.mm_use_im_start_end:
                conv.append_message(conv.roles[0],DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + message["content"])
            elif flag==1:
                conv.append_message(conv.roles[0],DEFAULT_IMAGE_TOKEN + '\n' + message["content"])
            else:
                conv.append_message(conv.roles[0], message["content"])
            image = None
        elif message['role'] == "assistant":
            conv.append_message(conv.roles[1], message["content"]) 
        flag+=1

    conv.append_message(conv.roles[1], None)
    #import pdb;pdb.set_trace()
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip("<s> ").strip("</s>").strip()
  #  conv.messages[-1][-1] = outputs

        # if args.debug:
        #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return outputs 

