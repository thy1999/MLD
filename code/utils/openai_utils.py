import tiktoken
from transformers import LlamaForCausalLM, AutoTokenizer

model2max_context = {
    "gpt-4": 7900,
    "gpt-4-0314": 7900,
    "llama2":4096,
    "gpt-3.5-turbo-0301": 3900,
    "gpt-3.5-turbo": 3900,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "vicuna":4096
}

class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    hf_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/llama-main/llama-2-7b-chat'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    inputs = tokenizer(string, return_tensors="pt")
    num_tokens = len(inputs['input_ids'][0])
    # encoding = tiktoken.encoding_for_model(model_name)
    # num_tokens = len(encoding.encode(string))
    return num_tokens

