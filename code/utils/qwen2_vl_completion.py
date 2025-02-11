

from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def qwen2_vl_generate(qwen2_vl_processor,
                  qwen2_vl_model, 
                  image_file,
                  messages,
                  temperature: float = 0.2,
                  max_new_tokens: int = 1024,
                ):
    history=[]
    flag=0
    for message in messages[0]:
        if message['role'] == 'system':
            system = message['content']
        elif message['role'] == "user":
            if flag==1 :
                message["content"] = [
                    {'type': 'video','video':image_file}, 
                    {'type': 'text','text': message['content']},
                ]
            
        elif message['role'] == "assistant":
            tuple_b = message["content"]
        history.append(message)
        flag+=1 
    text =qwen2_vl_processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)
    inputs = qwen2_vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(qwen2_vl_model.parameters()).device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    #import pdb;pdb.set_trace()
    generated_ids = qwen2_vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = qwen2_vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text[0]
