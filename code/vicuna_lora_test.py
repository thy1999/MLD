
import time
from pathlib import Path
import os
import json
import random
# random.seed(0)
import argparse
from utils.agent_lora import Agent
from datetime import datetime
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer
import ast
import torch
from utils.llama_chat_completion_lora import initialize_Llama
from utils.llama.tokenizer import Tokenizer
from utils.llama.model import ModelArgs, Transformer
import transformers
from utils.vicuna.fastchat.modules.gptq import GptqConfig
from utils.vicuna.fastchat.modules.awq import AWQConfig

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

import re
from collections import defaultdict
import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/Baichuan2-main/fine-tune_cp")
from core.utils import AverageMeter
from terminaltables import AsciiTable

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

import time
import random
#from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError

from transformers import LlamaForCausalLM, AutoTokenizer
from utils.llama_chat_completion_lora import Llama_generate
from utils.vicuna.fastchat.model.model_adapter import (
    get_conversation_template,
    get_generate_stream_function
)
from utils.vicuna.fastchat.model.model_codet5p import generate_stream_codet5p
from utils.llama_chat_completion_lora import create_dialogue_prompt


def load_model(base_model, model,lora_weights,load_8bit) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """

    if not base_model:
        raise ValueError(f'can not find base model name by the value: {model}')

    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')


    if model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if device == "cuda":
        if model == 'LLaMA-7B':
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            ) 
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            ) 
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"":0}
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

    return tokenizer, model


   
def stream_output( output_stream):
    pre = 0
    SS = ""
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            SS+=(" ".join(output_text[pre:now]))
            # print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    # print(" ".join(output_text[pre:]), flush=True)
    return " ".join(output_text)



device="cuda:1"
vicuna_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/FastChat-main/vicuna' 
vicuna_lora_weights = '/public/home/dzhang/pyProject/hytian/ZModel/LLM-Adapters/trained_models/vicuna_7B_ptb_all'
vicuna_tokenizer, vicuna_model = load_model(base_model=vicuna_model_path, model='Vicuna-7B',lora_weights=vicuna_lora_weights ,load_8bit='True')


messages=[[
            {
                "role": "system",
                "content": 'You are a debater. Welcome to the component syntax analysis competition, which will be conducted in a debate format. It\'s not necessary to fully agree with each other\'s perspectives, as our objective is to find the correct component syntax analysis result. Component syntax analysis aims to dissect sentences into their constituent parts and represent it as a hierarchical structure of bracketing. You need to identify various components in the sentence (such as noun phrases, verb phrases, Prepositional phrase, etc.) and combine these components together. Each element within the bracketing \'(\' and \')\' should represent either a single word or a grouped phrase. The debate topic is stated as follows: What is the correct constituency parsing result of the following sentence: "now add the beef 2 tbsp of flour 1 tsp of paprika 1 tbsp of tomato puree 2 bay leaves and 300ml beef stock" ?'
            },
            {
                "role": "user",
                "content": " Please perform component syntax analysis on the following sentences: 'now add the beef 2 tbsp of flour 1 tsp of paprika 1 tbsp of tomato puree 2 bay leaves and 300ml beef stock' -->"
            },
            {
                "role": "assistant",
                "content": '(now add ((the beef) (((2 tbsp) (of flour)) ((1 tsp) (of paprika)) ((1 tbsp) (of (tomato puree))) ((2 bay leaves) and (300ml beef stock)))))'
            },
            {
                "role": "user",
                "content": 'I provide the component syntax analysis result of the sentence that I believe is correct. However, You disagree with my result. Therefore, please provide the answer you think is right, and output it in a nested bracketing structure as follows: ((A (B C)) (D E)). The output must be a nested bracketing structure without any extra content, containing all the words of the sentence, and the number of "(" must match the number of ")". Please strictly adhere to the nested bracketing structure and do not output any irrelevant content. Here are some examples for your reference and learning. Please format the output according to the examples.\n\nSentence: "The children ate the cake with a spoon"\nMy_result: (The (children ((ate (the cake)) (with (a spoon)))))\nYour_result: ((The children) ((ate (the cake)) (with (a spoon))))\n\nSentence: "The little boy likes red tomatoes"\nMy_result:(The (little boy) (likes (red tomatoes)))\nYour_result: ((The little boy) (likes (red tomatoes)))\n\nSentence: "One man wrapped several diamonds in the knot of his tie"\nMy_result: ((One man wrapped (several diamonds) (in (the knot) (of (his tie)))))\nYour_result: ((One man) (wrapped (several diamonds) (in (the knot) (of (his tie)))))\n\nSentence: "now add the beef 2 tbsp of flour 1 tsp of paprika 1 tbsp of tomato puree 2 bay leaves and 300ml beef stock"\nMy_result: (((now add) ((the beef) (((2 tbsp) (of flour)) ((1 tsp) (of paprika)) ((1 tbsp) (of (tomato puree))) ((2 bay leaves) and (300ml beef stock))))))\nYour_result:'
            }
        ]]

vicuna_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/FastChat-main/vicuna'
conv = get_conversation_template(vicuna_model_path)
for message in messages[0]:
    if message['role'] == 'system':
        conv.set_system_message(message['content'])
    elif message['role'] == "user":
        conv.append_message(conv.roles[0], message['content'])
    elif message['role'] == "assistant":
        conv.append_message(conv.roles[1], message["content"])
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
generate_stream_func = get_generate_stream_function(vicuna_model, vicuna_model_path)
gen_params = {
        "model": vicuna_model_path,
        "prompt": prompt,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        # "stop": conv.stop_str,
        # "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
output_stream = generate_stream_func(
        vicuna_model,
        vicuna_tokenizer,
        gen_params,
        device= "cuda:1",
        context_len= 4096,
        judge_sent_end= True,
    )

gen = stream_output(output_stream)
print(gen)