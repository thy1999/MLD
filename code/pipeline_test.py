from transformers import AutoTokenizer
import transformers
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Literal, Optional, Tuple, TypedDict

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
bos, eos="<s>","</s>"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

def create_dialogue_prompt(dialogs,max_gen_len):
    if max_gen_len is None:
            max_gen_len = 4096 - 1
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[str] = [
            f"{bos}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}{eos} " 
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ]
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens +=[f"{bos}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}{eos}"]
        prompt_tokens.append(dialog_tokens)
    return "".join(prompt_tokens[0])

messages=[[
            {
                "role": "system",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
            },
            {
                "role": "user",
                "content": """The sentence is as follows: 'now add the beef 2 tbsp of flour 1 tsp of paprika 1 tbsp of tomato puree 2 bay leaves and 300ml beef stock', 
                and the component syntax analysis are as follows:
               (now add ((the beef) (((2 tbsp) (of flour)) ((1 tsp) (of paprika)) ((1 tbsp) (of (tomato puree))) ((2 bay leaves) and (300ml beef stock))))). 
                Some noun phrases and verb phrases are missing from the component syntax analysis result. Please reperform component syntax analysis on the sentences:'now add the beef 2 tbsp of flour 1 tsp of paprika 1 tbsp of tomato puree 2 bay leaves and 300ml beef stock'-->"""
            }
        ]]
h=create_dialogue_prompt(messages,4096)

base_model = '/public/home/dzhang/pyProject/hytian/ZModel/llama-main/llama-2-7b-chat-hf' 
lora_weights = '/public/home/dzhang/pyProject/hytian/ZModel/LLM-Adapters/trained_models/llama2_7B_ptb_all'
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) 
tokenizer = AutoTokenizer.from_pretrained(base_model)
pipeline=transformers.pipeline("text-generation",model=base_model,tokenizer=tokenizer,torch_dtype=torch.float16,device_map="auto",)
pipeline.model =PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )

import pdb;pdb.set_trace()
#sequences = pipeline(h,do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=2048,)
sequences = pipeline(
    h,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2048,
)

print("sequences=", sequences)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
#print([seq['generated_text'] for seq in sequences])
