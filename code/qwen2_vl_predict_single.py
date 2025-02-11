


import time
import copy
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
# from utils.llama_chat_completion_lora import initialize_Llama
# from utils.llama.tokenizer import Tokenizer
# from utils.llama.model import ModelArgs, Transformer
# import transformers
# from utils.vicuna.fastchat.modules.gptq import GptqConfig
# from utils.vicuna.fastchat.modules.awq import AWQConfig

# from fairscale.nn.model_parallel.initialize import (
#     get_model_parallel_rank,
#     initialize_model_parallel,
#     model_parallel_is_initialized,
# )

#qwen2-vl
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


#qwen
import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/YModel/Qwen-VL-master")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM


#llava
import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/YModel/LLaVA-main")
import argparse
import torch

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


import signal
import time


import time
import random
#from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError

from transformers import LlamaForCausalLM, AutoTokenizer
from utils.llava_chat_completion import Llama_generate  #导入llava评估的方法
from utils.qwen_chat_completion import Qwen_generate  #导入llava评估的方法
from utils.qwen2_vl_completion import qwen2_vl_generate


#把span变成（）
def spans_to_tree(sentence, spans):
    words = sentence.split()
    length=len(words)
    spans = set(tuple(span) for span in spans)
    spans = list(list(span) for span in spans)
    unique_spans = sorted(spans, key=lambda x: (x[1], -x[0]))
    tree = [(i, "") for i in range(length)]
    tree = dict(tree)
    result=""

    for i in range(len(unique_spans)):
        l, r = unique_spans[i]
        tree[l] = "(" + tree[l]
        tree[r] = tree[r] + ")"
    for j in range(length):
        if "(" in tree[j]:
            result=result+tree[j]+words[j]+" "
        elif ")" in tree[j]:
            result=result+words[j]+tree[j]+" "
        else:
            result=result+words[j]+" "
    return  result.strip()

#把()变成[]
def small_to_mid_transform(sentence):
    ### ()转换为[]
    stack = []
    current = []
    word = ''
    for char in sentence:
        if char == '(':
            if word:
                current.append(word)
                word = ''
            stack.append(current)
            current = []
        elif char == ')':
            if word:
                current.append(word)
                word = ''
            popped = stack.pop()
            popped.append(current)
            current = popped
        elif char == ' ':
            if word:
                current.append(word)
                word = ''
        else:
            word += char
    if word:
        current.append(word)
    return current[0]

#把（）变成span，无重复
def extract_spans_from_tree(tree, words):
    """
    Extract spans from a constituency parse tree based on the original sentence word indexes.
    
    Args:
    tree (str): A constituency parse tree as a string.
    words (list): The original sentence split into words.
    
    Returns:
    list: The list of extracted spans.
    """
    def tokenize_tree(tree):
        """
        Tokenize the tree into brackets, words, and spaces.
        """
        tokens, token = [], ""
        for char in tree:
            if char in "()":
                if token:
                    tokens.append(token)
                    token = ""
                tokens.append(char)
            elif char == " ":
                if token:
                    tokens.append(token)
                    token = ""
            else:
                token += char
        if token:
            tokens.append(token)
        return tokens

    def find_span(tokens):
        """
        Find the span of each constituent in the tree.
        """
        stack, spans, word_index = [], [], 0
        for token in tokens:
            if token == '(':
                stack.append(word_index)
            elif token == ')':
                start = stack.pop()
                spans.append([start, word_index-1])
            else:
                # Increment word index if token is not a bracket and is in the words list
                if token in words:
                    word_index += 1
        return spans

    def remove_duplicates(lst):
        # 使用集合去重，先将内部列表转换为元组
        unique_tuples = set(tuple(x) for x in lst)
        # 将去重后的元组转换回列表
        unique_lists = [list(x) for x in unique_tuples]
        return unique_lists
    # Tokenize the tree and find spans
    tokens = tokenize_tree(tree)
    spans = find_span(tokens)
    return remove_duplicates(spans)

#把（）变成span,有重复
def extract_spans_from_tree_hh(tree, words):
    """
    Extract spans from a constituency parse tree based on the original sentence word indexes.
    
    Args:
    tree (str): A constituency parse tree as a string.
    words (list): The original sentence split into words.
    
    Returns:
    list: The list of extracted spans.
    """
    def tokenize_tree(tree):
        """
        Tokenize the tree into brackets, words, and spaces.
        """
        tokens, token = [], ""
        for char in tree:
            if char in "()":
                if token:
                    tokens.append(token)
                    token = ""
                tokens.append(char)
            elif char == " ":
                if token:
                    tokens.append(token)
                    token = ""
            else:
                token += char
        if token:
            tokens.append(token)
        return tokens

    def find_span(tokens):
        """
        Find the span of each constituent in the tree.
        """
        stack, spans, word_index = [], [], 0
        for token in tokens:
            if token == '(':
                stack.append(word_index)
            elif token == ')':
                start = stack.pop()
                spans.append([start, word_index-1])
            else:
                # Increment word index if token is not a bracket and is in the words list
                if token in words:
                    word_index += 1
        return spans
   
    # Tokenize the tree and find spans
    tokens = tokenize_tree(tree)
    spans = find_span(tokens)
    return spans

#把[]变成（）
def to_parentheses_format(components):
    if isinstance(components, str):
        return components
    else:
        inner_parts = [to_parentheses_format(c) for c in components]
        return '(' + ' '.join(inner_parts) + ')'

#把（）变成成分
def extract_constituents_from_tree(tree, words):
    """
    Extract spans from a constituency parse tree based on the original sentence word indexes.
    
    Args:
    tree (str): A constituency parse tree as a string.
    words (list): The original sentence split into words.
    
    Returns:
    list: The list of extracted spans.
    """
    def tokenize_tree(tree):
        """
        Tokenize the tree into brackets, words, and spaces.
        """
        tokens, token = [], ""
        for char in tree:
            if char in "()":
                if token:
                    tokens.append(token)
                    token = ""
                tokens.append(char)
            elif char == " ":
                if token:
                    tokens.append(token)
                    token = ""
            else:
                token += char
        if token:
            tokens.append(token)
        return tokens

    def find_span(tokens):
        """
        Find the span of each constituent in the tree.
        """
        stack, spans, word_index = [], [], 0
        for token in tokens:
            if token == '(':
                stack.append(word_index)
            elif token == ')':
                start = stack.pop()
                spans.append([start, word_index-1])
            else:
                # Increment word index if token is not a bracket and is in the words list
                if token in words:
                    word_index += 1
        return spans
    
    def remove_duplicates(lst):
        # 使用集合去重，先将内部列表转换为元组
        unique_tuples = set(tuple(x) for x in lst)
        # 将去重后的元组转换回列表
        unique_lists = [list(x) for x in unique_tuples]
        return unique_lists
    # Tokenize the tree and find spans

    def spans_to_words(words, spans):
        # 存储转换后的单词跨度
        words_from_spans = []

        for span in spans:
            # 跨度开始和结束
            start, end = span
            # 提取跨度对应的单词序列，使用join连接成字符串
            span_words = ' '.join(words[start:end+1])
            words_from_spans.append(span_words)

        return words_from_spans


    tokens = tokenize_tree(tree)
    spans = find_span(tokens)
    unsort_spans=remove_duplicates(spans)
    sorted_spans=sorted(unsort_spans, key=lambda x: (x[0], x[1]))
    return spans_to_words(words, sorted_spans)


def extract_words(nested_list):
    words = []
    # 定义一个递归函数来处理嵌套列表
    def extract_words_recursive(nested):
        for item in nested:
            if isinstance(item, list):
                extract_words_recursive(item)
            elif isinstance(item, str):
                # 根据空格分割单词
                words.extend(item.split())
    extract_words_recursive(nested_list)
    return words

# def Parentheses_Match(text):
#     ### 确保中括号匹配。
#     stack = []
#     for char in text:
#         if char == '[':
#             stack.append(char)
#         elif char == ']':
#             if stack:
#                 stack.pop()
#             else:
#                 # Ideally, this situation should not occur if input is expected to have more '[' than ']'
#                 text = '[' + text  # Prepend a missing '['
#     # Append missing ']' for each unmatched '['
#     text += ']' * len(stack)
#     return text

 ### 根据标签找到对应的文本并进行提取。
def Find_Start_End_with_Label(ori_str, label):
    if label == "mid":
        return ori_str[ori_str.find("["):ori_str.rfind("]")+1]
    elif label == "big":
        return ori_str[ori_str.find("{"):ori_str.find("}")+1].split(",")[0]+', "constituency parsing result": "None"}'

# 检查每个字符，如果是单引号或双引号，并且该引号的后面没有紧跟着另一个引号，则进行修复
def fix_missing_quotes(s):
    fixed_string = ''
    fuhao=""
    in_quotes = False
    index0=-1
    count=0
    for index,char in enumerate(s):
        if char=='{' :
            index0=index
        if s[index-1]==":" or s[index-1]==",":
            index0=index-1
        if char == "'" or char == '"':
            if not in_quotes:
                in_quotes = True
            else:
                in_quotes = False
            fixed_string += char
            fuhao += char 
        elif in_quotes and char == ':':  # 处理缺失的引号
            if fixed_string[-1]!=fuhao[-1]:
                fixed_string += fuhao[-1]  # 检查前一个字符的引号类型
            else:
                fixed_string =  fixed_string[:index0+count+1] + fuhao[-1] + fixed_string[index0+count+1:] 
            count=count+1
            fixed_string += char
            fuhao += char
            in_quotes = False
        elif in_quotes and char == '}':  # 处理缺失的引号
            if fixed_string[-1]!=fuhao[-1]:
                fixed_string += fuhao[-1]  # 检查前一个字符的引号类型
            else:
                fixed_string =  fixed_string[:index0+count+1] + fuhao[-1] + fixed_string[index0+count+1:] 
            count=count+1
            fixed_string += char
            fuhao += char
            in_quotes = False
        elif in_quotes and char ==',':  # 处理缺失的引号
            if fixed_string[-1]!=fuhao[-1]:
                fixed_string += fuhao[-1]  # 检查前一个字符的引号类型
            else:
                fixed_string =  fixed_string[:index0+count+1] + fuhao[-1] + fixed_string[index0+count+1:] 
            count=count+1
            fixed_string += char
            fuhao += char
            in_quotes = False
        else:
            fixed_string += char
    # 如果最后一个字符是引号，但没有配对的引号，补全引号
    if in_quotes:
        fixed_string += fuhao[-1]
        in_quotes = False
    return fixed_string


def Parentheses_Match(text):
    ### 确保小括号匹配。
    stack = []
    for char in text:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                # Ideally, this situation should not occur if input is expected to have more '[' than ']'
                text = '(' + text  # Prepend a missing '['
    # Append missing ']' for each unmatched '['
    text += ')' * len(stack)
    return text

def extract_words(nested_list):
    words = []
    # 定义一个递归函数来处理嵌套列表
    def extract_words_recursive(nested):
        for item in nested:
            if isinstance(item, list):
                extract_words_recursive(item)
            elif isinstance(item, str):
                # 根据空格分割单词
                words.extend(item.split())
    extract_words_recursive(nested_list)
    return words


def match_words(sentence, pred):
    ### 单词匹配，保证原始句子中的单词与嵌套列表中的单词一致。
    nested_lst = small_to_mid_transform(pred)
    assert type(nested_lst)==list
    nested_str = str(nested_lst)
    words_list = sentence.split()
    st = 0
    new_s = ""
    i = 0
    while i<len(nested_str):
        tp = nested_str[i]
        if tp=="'" or tp=='"':
            j = i+1
            if tp == "'":
                while (j<len(nested_str) and nested_str[j]!="'"):
                    j+=1
            elif tp == '"':
                while (j<len(nested_str) and nested_str[j]!='"'):
                    j+=1
            nest_word = nested_str[i+1:j]
            if st < len(words_list):
                if nest_word == words_list[st]:
                    new_s+=nested_str[i:j+1]
                    st+=1
                    i=j+1
                elif words_list[st] in nest_word:
                    tpp = []
                    while st<len(words_list) and words_list[st] in nest_word:
                        if "'" in words_list[st]:
                            tpp.append('\"'+words_list[st]+'\"')
                        else:
                            tpp.append("'"+words_list[st]+"'")
                        st+=1
                    if nested_str[i-1]=="[" or nested_str[i-1]=="," or nested_str[i-2:i]==", ":
                        new_s+=(", ".join(tpp))
                    else:
                        new_s+=(", "+", ".join(tpp))
                    i=j+1
                else:
                    last = st
                    st += 1 
                    while st<len(words_list) and nest_word!=words_list[st]:
                        st += 1
                    if st>=len(words_list):
                        st = last
                        if j+1<len(nested_str):
                            if nested_str[j+1]==",":
                                i = j+2
                            elif nested_str[j+1]=="]":
                                # if nested_str[i-1]==" ":
                                #     new_s = new_s[:-2]
                                i = j+1
                    else:                        
                        for k in range(last,st):
                            new_s+=("'"+str(words_list[k]).replace("\\","\\\\").replace("'","\\'")+"', ")
                        new_s+=nested_str[i:j+1]
                        st+=1
                        i=j+1
            else:
                    i = j+1
        else:  
            if st>=len(words_list) and tp!=']':
                new_s=new_s.strip(",")
                count_lf=new_s.count("[")
                count_rf=new_s.count("]")
                if count_lf>count_rf:
                    new_s=new_s + (count_lf- count_rf)*']'
                break
            new_s += tp
            i += 1
    # if sentence == "\ it started in 1882 with the Winnipeg Fire Department and grew from there .":
    #     #import pdb;pdb.set_trace()
    if st!=len(words_list):
        new_s = str(eval(new_s))
        ind = new_s.rfind("'")
        ind2 = new_s.rfind('"')
        ind = max(ind,ind2)
        temp_list = []
        for k in words_list[st:len(words_list)]:
            if "'" in k:
                temp_list.append('\"'+k+'\"')
            else:
                temp_list.append("'"+k+"'")
        temp = ", ".join(temp_list)
        if ind==-1:
            new_s = "["+temp+"]"
        else:
            new_s = new_s[:ind+1]+", "+temp+new_s[ind+1:]
    new_s = eval(new_s.replace('[],', '').replace('[]',''))
    assert extract_words(new_s)==words_list
    return to_parentheses_format(new_s)

def Format_output(sentence, pred_out):
    ### 输出标准化。
    prefix_word = sentence.split()[0]
    second_word = sentence.split()[1]
    third_word = sentence.split()[2]
    pattern1 = re.compile(re.escape(prefix_word) + r'\s*\)*\s*\(*\s*' + re.escape(second_word), re.IGNORECASE)
    pattern2 = re.compile(re.escape(prefix_word), re.IGNORECASE)
    pattern3 = re.compile(re.escape(second_word), re.IGNORECASE)
    pattern4 = re.compile(re.escape(second_word) + r'\s*\)*\s*\(*\s*' + re.escape(third_word), re.IGNORECASE)
    match1 = pattern1.search(pred_out)
    match2 = pattern2.search(pred_out)
    match3 = pattern3.search(pred_out)
    match4 = pattern4.search(pred_out)
    lst1 = pattern1.split(pred_out, 1)[1:]
    lst2 = pattern2.split(pred_out, 1)[1:]
    lst3 = pattern3.split(pred_out, 1)[1:]
    lst4 = pattern4.split(pred_out, 1)[1:]
    pred_out=''.join(lst1)
    pred_out4=''.join(lst4)
    if match1: #匹配第一第二个单词
        pred_out='('+match1.group(0).replace(match2.group(0),prefix_word).replace(match3.group(0),second_word) +pred_out
    elif match4: #匹配第二第三个单词
        pred_out='('+prefix_word+' '+match4.group(0).replace(match3.group(0),second_word) +pred_out4
    # elif match2: #匹配第一个单词
    #     pred_out='('+match2.group(0).replace(match2.group(0),prefix_word) +pred_out2
    pred_out = pred_out.replace("1. (","(").replace("'[","[")
    response_list = ["No.","Yes.","Okay."]
    if pred_out in response_list or pred_out[:-1]==sentence:
        pred_out = "("+sentence+")"
    if len(sentence.split())+50<len(pred_out.split()):
        if "Perform constituency parsing on the following sentence:" in pred_out:
            pred_out = "("+sentence+")"
        else:
            pred_out = " ".join(pred_out.split()[:len(sentence.split())])
    if "'\"" in pred_out and "'\"" not in sentence:
        pred_out=pred_out.replace("'\"",'"')
    if pred_out.count(".)")>10:
        if "." not in sentence:
            pred_out = pred_out.replace(".)","")
        elif "." in sentence:
            ind = pred_out.find(".)")
            pred_out = pred_out[:ind+2]+pred_out[ind+2:].replace(".)","")
    bracket_flag = 0
    if "]" in pred_out and "]" in sentence:
        pred_out = pred_out.replace("]","##bracket##")
        sentence = sentence.replace("]","##bracket##")
        bracket_flag = 1
    if pred_out[0]!="(":
        pred_out = "("+pred_out+")"
    pred_text = Parentheses_Match(pred_out)
    # seq_len=len(sentence.split())
    if pred_text.replace("(","").replace(")","")==sentence or " ".join(extract_words(small_to_mid_transform(pred_text)))==sentence:
        spans_list=extract_spans_from_tree(pred_text,sentence.split()) #利用span列表去重
        spans_list=[sublist for sublist in spans_list if sublist[0] != sublist[1]]
        pred_text = spans_to_tree(sentence, spans_list)
        if bracket_flag:
            pred_text = pred_text.replace("##bracket##","]")
        # count = spans_list.count([0, seq_len-1])
        # while count > 1:
        #     pred_text = pred_text[1:-1]   # 去除字符串两端的括号
        #     count -= 1
        # if count == 1:
        #     return pred_text
        return pred_text
    else:
        pred_new = match_words(sentence, pred_text)
        spans_list=extract_spans_from_tree(pred_new,sentence.split()) #利用span列表去重
        spans_list=[sublist for sublist in spans_list if sublist[0] != sublist[1]]
        pred_new = spans_to_tree(sentence, spans_list)
        if bracket_flag:
            pred_new = pred_new.replace("##bracket##","]")
        # count = spans_list.count([0, seq_len-1])
        # while count > 1:
        #     pred_new = pred_new[1:-1]   # 去除字符串两端的括号
        #     count -= 1
        # if count == 1:
        #     return pred_new
 
        return pred_new



def find_keyword_values(data):
    result = {}
    for key, value in data.items():
        if 'whether' in key.lower() :
            if value.lower()=="yes":
                result['whether there is a preference']="Yes"
            else:
                result['whether there is a preference']="No"
        elif 'supported' in key.lower():
            if value.lower()=="affirmative":
                result['supported side']="Affirmative"
            elif value.lower()=="negative":
                result['supported side']="Negative"
            else:
                result['supported side']="None"
        elif 'result' in key.lower():
            result['constituency parsing result']= value
        elif 'reason' in key.lower():
            result['reason']=value
        else:
            result[key]= value
    return result


def initialize_stats():
    per_label_f1 = defaultdict(list)
    by_length_f1 = defaultdict(list)
    sent_f1, corpus_f1 = AverageMeter(), [0., 0., 0.]
    return per_label_f1, by_length_f1, sent_f1, corpus_f1

def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn


def get_batch_stats(lengths, pred_spans, gold_spans, labels):
    
    per_label_f1, by_length_f1, sent_f1, corpus_f1 = initialize_stats()
    for max_len, pred_span, gold_span, label in zip(lengths, pred_spans, gold_spans, labels):
        pred_set = set((a[0], a[1]) for a in pred_span if a[0] != a[1] and a != [0, max_len-1])
        gold_set = set((a[0], a[1]) for a in gold_span if a[0] != a[1] and a != [0, max_len-1])
        if len(gold_set) == 0:
            continue
        tp, fp, fn = get_stats(pred_set, gold_set)
        corpus_f1[0] += tp
        corpus_f1[1] += fp
        corpus_f1[2] += fn
        overlap = pred_set.intersection(gold_set)
        prec = float(len(overlap)) / (len(pred_set) + 1e-8)
        reca = float(len(overlap)) / (len(gold_set) + 1e-8)

        if len(gold_set) == 0:
            reca = 1.
            if len(pred_set) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        sent_f1.update(f1)

        for l, gs in zip(label, gold_span):
            if gs[0] == gs[1] or gs == [0, max_len - 1]:
                continue
            l = re.split("=|-", l)[0]
            per_label_f1.setdefault(l, [0., 0.])
            per_label_f1[l][0] += 1

            lspan = gs[1] - gs[0] + 1
            by_length_f1.setdefault(lspan, [0., 0.])
            by_length_f1[lspan][0] += 1

            if tuple(gs) in pred_set:
                per_label_f1[l][1] += 1
                by_length_f1[lspan][1] += 1

    return per_label_f1, by_length_f1, sent_f1, corpus_f1

def get_f1s(stats):
    per_label_stat, by_length_stat, sent_stat, corpus_stat = stats


    f1s = {'C-F1': get_corpus_f1(corpus_stat), 'S-F1': sent_stat.avg}
    for k in ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP"]:
        if k in per_label_stat:
            f1s[k] = per_label_stat[k][1] / per_label_stat[k][0]
        else:
            f1s[k] = 0
    for k in by_length_stat.keys():
        if by_length_stat[k][0] >= 5:
            f1s[k] = by_length_stat[k][1]/by_length_stat[k][0]
    return f1s

def get_corpus_f1(stat):
    prec = stat[0] / (stat[0] + stat[1]) if stat[0] + stat[1] > 0 else 0.
    recall = stat[0] / (stat[0] + stat[2]) if stat[0] + stat[2] > 0 else 0.
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    return f1

def display_f1s(f1s, title=None, display_by_length=False):
    if display_by_length:
        column_names = list(range(2,16))
    else:
        column_names = ["NP", "VP", "PP", "SBAR", "ADJP", "ADVP", "C-F1", "S-F1"]
    display_data = [column_names]
    display_data.append(['{:.02f}'.format(f1s[k]*100) for k in column_names])
    table = AsciiTable(display_data, title)
    for i in range(2):
        table.justify_columns[i] = 'center'
    return table.table


def evalute(max_lengths,pred_spans, gold_spans, gold_labels):
    stats = get_batch_stats(max_lengths,pred_spans, gold_spans, gold_labels)
    f1s = get_f1s(stats)
    result = display_f1s(f1s, 'performance on testing set')






 #读取qwen模型结果
model_dir = "/public/home/dzhang/pyProject/hytian/XModel/Qwen2-VL/output/qwen2-vl-7b-instruct/qwen2-vl-7b-instruct/v0-20250105-222105/checkpoint-140-merged"
# Load the model in half-precision on the available device(s)
qwen2_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, device_map="auto", torch_dtype = torch.float16)
min_pixels = 256*28*28
max_pixels = 1280*28*28
qwen2_vl_processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)


llm_pred_spans = []
parser_pred_spans = []
gold_spans = [] 
gold_labels = []
max_lengths = []
problem_data = []
llm_pred_data = []

save_file='/public/home/dzhang/pyProject/hytian/XModel/Multi-Agents-Debate-main/data/qwen2_vl_predict_youcook2_newv0.json'
test_gold_caps_path='/public/home/dzhang/pyProject/hytian/XModel/Qwen2-VL/youcookii_text_to_video_last.json'


with open(test_gold_caps_path,'r',encoding='utf-8') as file:
    line_cnt=-1
    output_data=[]
    for line in file:
        line_cnt+=1
        data = json.loads(line)
        # 获取 id 键对应的值
        id_value = data.get('id')
        
        # 获取另一个键对应的值（假设这个键为 'other_key'）
        sentence = data.get("caption")
        gold_span = data.get("span")
        gold_label = data.get("label")
        image_file = data.get('video_path')

        old_data={}
        old_data['sentence']=sentence
        old_data['label']=gold_label
        old_data['gold_span']=gold_span
      
        #s_prompt=""
        s_prompt="You are a debater. Welcome to the constituency parsing competition, which will be conducted in a debate format. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct constituency parsing result. constituency parsing aims to dissect sentences into their constituent parts and represent it as a hierarchical structure of bracketing. You need to identify various constituencies in the sentence (such as noun phrases, verb phrases, Prepositional phrase, etc.) and combine these constituencies together. Each element within the bracketing '(' and ')' should represent either a single word or a grouped phrase. The debate topic is stated as follows: What is the correct constituency parsing result of the following sentence: \"##sentence##\" ?"
        u_prompt="Please perform constituency parsing on the following description:\nDescription: ##sentence##.\nConstituency Parsing:"
        messages=[[{'role':'system','content':s_prompt.replace('##sentence##', old_data['sentence'])},{'role':"user",'content':u_prompt.replace('##sentence##', old_data['sentence'])}]]
     
        #import pdb;pdb.set_trace()
        cnt=0
        while True:
            try:
                cnt+=1
                if cnt>3:
                    llm_pred_tree='('+old_data['sentence']+')'
                    break
                gen=qwen2_vl_generate(qwen2_vl_processor,
                  qwen2_vl_model, 
                  image_file,
                  messages,
                  temperature= 0.2,
                  max_new_tokens = 1024,
                )  
                #print(gen)  
               # import pdb;pdb.set_trace()
                try:
                    llm_pred_tree = Format_output(old_data['sentence'], gen.split('the constituency parsing result for the sentence:\n')[-1].split('\n')[0])
                except Exception as e:
                    llm_pred_tree = Format_output(old_data['sentence'], gen.split('\n\n')[1].split('\n')[1])
                if len(old_data['sentence'].split())<3:
                    break
                if llm_pred_tree[1:-1] ==old_data['sentence']:
                    llm_pred_tree = Format_output(old_data['sentence'], gen.split('\n')[0])
                if llm_pred_tree[1:-1] ==old_data['sentence']:
                    llm_pred_tree = Format_output(old_data['sentence'], gen.split('\n')[-1])
                llm_pred_tree=llm_pred_tree.replace("()","")
                
                if "()" not in llm_pred_tree and llm_pred_tree[1:-1]!=old_data['sentence'] and llm_pred_tree!="":
                    break
            except Exception as e:
                    print("An error occurred:", e)
                    print("Trying again...")
        #import pdb;pdb.set_trace()

       
        old_data['output_pred'] = llm_pred_tree
        output_data.append(old_data)
        with open(save_file, 'w+') as f:
            for old_data in output_data:
                f.write(json.dumps(old_data) + '\n')


        llm_pred=extract_spans_from_tree(llm_pred_tree,  old_data['sentence'].split())
        llm_pred_spans.append(llm_pred)
        gold_spans.append(gold_span)
        gold_labels.append(gold_label)
        max_lengths.append(len(sentence.split()))
        
llm_stats = get_batch_stats(max_lengths, llm_pred_spans, gold_spans, gold_labels)
llm_f1s = get_f1s(llm_stats)
llm_result = display_f1s(llm_f1s, 'performance on llm testing set')  
print(llm_result)
