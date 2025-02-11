

import time
import random
#from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError

from transformers import LlamaForCausalLM, AutoTokenizer
from utils.llama_chat_completion import Llama_generate
from utils.vicuna.fastchat.model.model_adapter import (
    get_conversation_template,
    get_generate_stream_function
)
from utils.vicuna.fastchat.model.model_codet5p import generate_stream_codet5p

support_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0314', "llama2", "videollava", "vicuna", "llama2_base"]

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0) -> None:
        """Create an agent

        Args:
            model_name(str): model name
            name (str): name of this agent
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

    #@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError), max_tries=20)
    def query(self, messages: "list[dict]", temperature: float) -> str:
        """make a query

        Args:
            messages (list[dict]): chat history in turbo format
            max_tokens (int): max token in api call
            api_key (str): openai api key
            temperature (float): sampling temperature

        Raises:
            OutOfQuotaException: the apikey has out of quota
            AccessTerminatedException: the apikey has been ban

        Returns:
            str: the return msg
        """
        # time.sleep(self.sleep_time)
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
        if self.model_name in support_models:
            # num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
            # max_token = model2max_context[self.model_name] - num_context_token
            messages = [messages]
            if self.model_name == 'llama2':
                gen = Llama_generate(self.llama_generator, messages)
            elif self.model_name == 'vicuna':
                vicuna_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/FastChat-main/vicuna'
                conv = get_conversation_template(vicuna_model_path)
                for message in messages[0]:
                    if message['role'] == 'system':
                        conv.set_system_message(message['content'])
                    elif message['role'] == "user":
                        conv.append_message(conv.roles[0], message['content'])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                generate_stream_func = get_generate_stream_function(self.vicuna_model, vicuna_model_path)
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
                    self.vicuna_model,
                    self.vicuna_tokenizer,
                    gen_params,
                    device= "cuda:1",
                    context_len= 4096,
                    judge_sent_end= True,
                )
                gen = self.stream_output(output_stream)
            # inputs_text = ""
            # for i in self.memory_lst:
            #     inputs_text += i['content']
            # inputs = self.tokenizer(inputs_text, return_tensors="pt")
            # inputs= inputs.to('cuda')
            # generate_ids = self.model.generate(inputs.input_ids, max_length=max_token)
            # gen = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split('\n')[-1]
        else:
            import pdb;pdb.set_trace()
        return gen

    def set_meta_prompt(self, meta_prompt: str):
        """Set the meta_prompt

        Args:
            meta_prompt (str): the meta prompt
        """
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        """Add an new event in the memory

        Args:
            event (str): string that describe the event.
        """
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
        """
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def remove_last_memory(self):
        self.memory_lst = self.memory_lst[:-1]

    def ask(self, temperature: float=None):
        """Query for answer

        Args:
        """
        # query
        # num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        # max_token = model2max_context[self.model_name] - num_context_token
        return self.query(self.memory_lst, temperature=temperature if temperature else self.temperature)
        # return self.query(temperature=temperature if temperature else self.temperature)
    
    def stream_output(self, output_stream):
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


