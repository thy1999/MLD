import sys
sys.path.append("/Video-LLaVA-main")
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer



def videollava_generate(videollava_tokenizer,
                  videollava_model, 
                  videollava_processor,
                  image_file,
                  messages,
                  temperature: float = 0.2,
                  max_new_tokens: int = 2048,
                ):
    model_name = get_model_name_from_path('/Video-LLaVA-main/Video-LLaVA-7B')
    video_processor = videollava_processor['video']
    conv_mode = "llava_v1"
    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    video_tensor = video_processor(image_file, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(videollava_model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(videollava_model.device, dtype=torch.float16)
    flag=0
    for message in messages[0]:
        if message['role'] == 'system':
            conv.system = message['content']
        elif message['role'] == "user":
            if flag==1 :
                tuple_a = ' '.join([DEFAULT_IMAGE_TOKEN] * videollava_model.get_video_tower().config.num_frames) + '\n' + message['content']
            else:
                tuple_a = message["content"]
            conv.append_message(conv.roles[0], tuple_a)
        elif message['role'] == "assistant":
            tuple_b = message["content"]
            conv.append_message(conv.roles[1], tuple_b)
        flag+=1
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #import pdb;pdb.set_trace()
    input_ids = tokenizer_image_token(prompt, videollava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    #streamer = TextStreamer(videollava_tokenizer, skip_prompt=True, skip_special_tokens=True)
    stopping_criteria = KeywordsStoppingCriteria(keywords, videollava_tokenizer, input_ids)
    #import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = videollava_model.generate(input_ids,images=tensor,do_sample=True,temperature=temperature,max_new_tokens=max_new_tokens,use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = videollava_tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().strip('</s>')
    return outputs