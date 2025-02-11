import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main")

import argparse
import os

import torch

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer



def videollava_generate(videollava_tokenizer,
                  videollava_model, 
                  videollava_processor,
                  image_file,
                  messages,
                  temperature: float = 0.2,
                  max_new_tokens: int = 2048,
                ):
    model_name = get_model_name_from_path('/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main/Video-LLaVA-7B')
    image_processor, video_processor = videollava_processor['image'], videollava_processor['video']
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
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
    
    tensor = []
    special_token = []
    image_file = image_file if isinstance(image_file, list) else [image_file]
    for file in  image_file:
        if os.path.splitext(file)[-1].lower() in image_ext:
            file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(videollava_model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN]
        elif os.path.splitext(file)[-1].lower() in video_ext:
            file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(videollava_model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN] * videollava_model.get_video_tower().config.num_frames
        else:
            raise ValueError(f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
        #print(file.shape)
        tensor.append(file)




    # try:
    #     inp = input(f"{roles[0]}: ")
    # except EOFError:
    #     inp = ""
    # if not inp:
    #     print("exit...")
    #     break

    # print(f"{roles[1]}: ", end="")

    flag=0
    for message in messages[0]:
        if message['role'] == 'system':
            conv.system = message['content']
        elif message['role'] == "user":
            if flag==1 :
                # if file is not None:
                    # first message
                if getattr(videollava_model.config, "mm_use_im_start_end", False):
                    inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + message["content"]
                else:
                    inp = ''.join(special_token) + '\n' + message["content"]
                    # conv.append_message(conv.roles[0], inp)
                    # file = None
                # else:
                #     # later messages
                #     conv.append_message(conv.roles[0], inp)
            else:
                inp = message["content"]
            conv.append_message(conv.roles[0], inp)
        elif message['role'] == "assistant":
            tuple_b = message["content"]
            conv.append_message(conv.roles[1], tuple_b)
        flag+=1
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, videollava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(videollava_model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, videollava_tokenizer, input_ids)
    streamer = TextStreamer(videollava_tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = videollava_model.generate(
            input_ids,
            images=tensor,  # video as fake images
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            #streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = videollava_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    return outputs