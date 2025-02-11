

#qwen
import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/YModel/Qwen-VL-master")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM


def Qwen_generate(qwen_tokenizer,
                  qwen_model, 
                  image_file,
                  messages,
                  temperature: float = 0.2,
                  max_new_tokens: int = 512,
                ):
    history=[]
    flag=0
    for message in messages[0]:
        if message['role'] == 'system':
            system = message['content']
        elif message['role'] == "user":
            if flag==1 :
                tuple_a = qwen_tokenizer.from_list_format([
                    {'image': image_file}, # Either a local path or an url
                    {'text': message['content']},
                ])
            else:
                tuple_a = message["content"]
        elif message['role'] == "assistant":
            tuple_b = message["content"]
            history.append(  (tuple_a, tuple_b)  )
        flag+=1
    response, history = qwen_model.chat(qwen_tokenizer, query=message["content"], history=history, system=system)
    return response
