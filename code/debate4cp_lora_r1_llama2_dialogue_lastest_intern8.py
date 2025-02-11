"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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

#qwen
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM


import signal
import time

import sys
# 导入 log 模块目录
sys.path.append("/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main")
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


#intern2-vl
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


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

# 定义处理超时的函数
def handler(signum, frame):
    raise TimeoutError("运行超时")

NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Judge",
]


#把span,labels变成树
def spans_labels_to_tree(sentence, spans,labels):
    words = sentence.split()
    length=len(words)
   
    # 将 spans 和 labels 组合
    combined = list(zip(spans, labels))

    # 按照 spans 的排序规则进行排序
    sorted_combined = sorted(combined, key=lambda x: (x[0][1], -x[0][0]))

    # 解压缩回 spans 和 labels
    sorted_spans, sorted_labels = zip(*sorted_combined)

    # 转换为列表（可选）
    unique_spans  = list(sorted_spans)
    labels = list(sorted_labels)

    tree = [(i, "") for i in range(length)]
    tree = dict(tree)
    result=""

    for i in range(len(unique_spans)):
        l, r = unique_spans[i]
        tree[l] = "("+labels[i]+" " + tree[l]
        tree[r] = tree[r] + ")"
    for j in range(length):
        if "(" in tree[j]:
            result=result+tree[j]+words[j]+" "
        elif ")" in tree[j]:
            result=result+words[j]+tree[j]+" "
        else:
            result=result+words[j]+" "
    return  result.strip()

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
        if "{" in ori_str and "}" in ori_str:
            try:
                return ori_str[ori_str.find("{"):ori_str.find("}")+1].split(",")[0]+', "constituency parsing result": "None"}'
            except Exception as e:
                raise ValueError("Invalid output(Judge)")
                #return '{"supported side": "Negative","constituency parsing result": "None"}'
        elif "{" in ori_str and "}" not in ori_str:
            try:
                return ori_str[ori_str.find("{"):].split(",")[0]+', "constituency parsing result": "None"}'
            except Exception as e:
                raise ValueError("Invalid output(Judge)")
                #return '{"supported side": "Negative","constituency parsing result": "None"}'
        else:
            if "Affirmative" in ori_str:
                return '{"supported side": "Affirmative","constituency parsing result": "None"}'
            elif "Negative" in ori_str:
                return '{"supported side": "Negative","constituency parsing result": "None"}'
            else:
                #return '{"supported side": "Negative","constituency parsing result": "None"}'
                raise ValueError("Invalid output(Judge)")


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

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature:float,  sleep_time: float,  videollava_model=None, videollava_tokenizer=None, videollava_processor=None,intern2_vl_model=None,intern2_vl_tokenizer=None, image_file=None) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        # if self.model_name == "llama2":
        #     hf_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/llama-main/llama-2-7b-chat'
        # elif self.model_name== "videollava":
        #     hf_model_path = '/public/home/dzhang/pyProject/hytian/ZModel/Video-LLaVA-main/LanguageBind/Video-LLaVA-7B'
        # self.model = LlamaForCausalLM.from_pretrained(hf_model_path, device_map="auto")
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        # if model_name == 'llava1.5' or model_name == 'llava1_6':
        #     self.llava_model=llava_model
        #     self.llava_tokenizer=llava_tokenizer
        #     self.llava_image_processor=llava_image_processor
        #     self.llava_context_len=llava_context_len
        #     self.llava1_6_model=llava1_6_model
        #     self.llava1_6_tokenizer=llava1_6_tokenizer
        #     self.llava1_6_image_processor=llava1_6_image_processor
        #     self.llava1_6_context_len=llava1_6_context_len
        #     self.image_file=image_file
        # elif model_name == "qwen":
        #     self.qwen_model = qwen_model
        #     self.qwen_tokenizer = qwen_tokenizer
        #     self.image_file=image_file
        if model_name=='intern2_vl':
            self.intern2_vl_model = intern2_vl_model
            self.intern2_vl_tokenizer = intern2_vl_tokenizer
            self.image_file=image_file
        elif model_name=='videollava':
            self.videollava_model=videollava_model
            self.videollava_tokenizer=videollava_tokenizer
            self.videollava_processor=videollava_processor
            self.image_file=image_file
        



class Debate:
    def __init__(self,
            temperature: float=0, 
            num_players: int=3, 
            save_file_dir: str=None,
            prompts_path: str=None,
            max_round: int=4,
            sleep_time: float=0,
            videollava_model=None,
            videollava_tokenizer=None,
            videollava_processor=None,
            intern2_vl_model=None,
            intern2_vl_tokenizer=None,      
            image_file=None
        ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            save_file_dir (str): dir path to json file
            openai_api_key (str): As the parameter name suggests
            prompts_path (str): prompts path (json file)
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.videollava_model = videollava_model
        self.videollava_tokenizer = videollava_tokenizer
        self.videollava_processor = videollava_processor
        self.intern2_vl_model = intern2_vl_model
        self.intern2_vl_tokenizer = intern2_vl_tokenizer
        self.image_file =image_file

 

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'temperature': temperature,
            'num_players': num_players,
            'success': False,
            'sentence': '',
            "round_1 constituency parsing result": '',
            "round_2 constituency parsing result": '',
            "round_3 constituency parsing result": '',
            "round_4 constituency parsing result": '',
            "seq_len":"",
           # "Reason": '',
           # "supported side": '',
            'players': {},
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        # creat&init agents
        #import pdb;pdb.set_trace()
        self.creat_agents()
        self.init_agents()


    def init_prompt(self):
        def prompt_replace(key):
            self.save_file[key] = self.save_file[key].replace("##sentence##", self.save_file["sentence"])
        # prompt_replace("base_prompt")
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("model_prompt1")
        prompt_replace("affirmative_prompt")
        prompt_replace("negative_prompt")
        prompt_replace("debate_prompt")
        prompt_replace("judge_prompt_last")
        #prompt_replace("debate_prompt_round2")

    # def create_base(self, llama2_pred, vicuna_pred):

    #     self.save_file['llama2_pred'] = llama2_pred
    #     self.save_file['vicuna_pred'] = vicuna_pred


    def creat_agents(self):
        # creates players
        self.players = []

     
        self.players.append(DebatePlayer(model_name="videollava", name=NAME_LIST[0], temperature=self.temperature, sleep_time=self.sleep_time, videollava_model=self.videollava_model, videollava_tokenizer=self.videollava_tokenizer,videollava_processor=self.videollava_processor, image_file=self.image_file))
        self.players.append(DebatePlayer(model_name="intern2_vl", name=NAME_LIST[1], temperature=self.temperature, sleep_time=self.sleep_time,  intern2_vl_model=self.intern2_vl_model, intern2_vl_tokenizer=self.intern2_vl_tokenizer,image_file=self.image_file))
        self.players.append(DebatePlayer(model_name="videollava", name=NAME_LIST[2], temperature=self.temperature, sleep_time=self.sleep_time, videollava_model=self.videollava_model, videollava_tokenizer=self.videollava_tokenizer,videollava_processor=self.videollava_processor, image_file=self.image_file))
     
 
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.judge = self.players[2]


    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.negative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.judge.set_meta_prompt(self.save_file['moderator_meta_prompt'])
    
        
        # start: first round debate, state opinions

        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.save_file['model_prompt1'])
        flag=0
        cnt=0
        while True:
            try:
                flag=0
                cnt+=1
                if cnt>3:
                    self.aff_ans='('+self.save_file['sentence']+')'
                    break
                self.aff_ans1 = self.affirmative.ask()
                try:
                    self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('My_result')[-1].split('Your_result')[-1].split("correction:")[-1].split("correct result")[-1]) 
                except Exception as e:
                    self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[0])
                if len(self.save_file['sentence'].split())<3:
                    break
                if self.aff_ans[1:-1] ==self.save_file['sentence']:
                    self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[0])
                if self.aff_ans[1:-1] ==self.save_file['sentence']:
                    self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[-1])
                self.aff_ans=self.aff_ans.replace("()","")
                
                if "()" not in self.aff_ans and self.aff_ans[1:-1]!=self.save_file['sentence'] and self.aff_ans!="":
                    break
            except TimeoutError as e:
                flag=1
                if str(e) == "运行超时":
                    print("捕获到运行超时错误，任务被中断。")
                    break
            except Exception as e:
                    print("An error occurred:", e)
                    print("Trying again...")
        if flag==1:
            raise TimeoutError("运行超时")
        self.affirmative.add_memory(self.aff_ans)

     
        self.negative.add_event(self.save_file['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
        cnt=0
        while True:
            try:
                flag=0
                cnt+=1
                if cnt>3:
                    self.neg_ans='('+self.save_file['sentence']+')'
                    break
                self.neg_ans1 = self.negative.ask()
                #print(self.neg_ans1)
                #import pdb;pdb.set_trace()
                try:
                    self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('My_result: ')[-1].split('Your_result: ')[-1].split("correction:")[-1].split("correct result")[-1]) 
                except Exception as e:
                    self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[0])
                if len(self.save_file['sentence'].split())<3:
                    break
                if self.neg_ans[1:-1] ==self.save_file['sentence']:
                    self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[0])
                if self.neg_ans[1:-1] ==self.save_file['sentence']:
                    self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[-1])
                self.neg_ans=self.neg_ans.replace("()","")
                
                if "()" not in self.neg_ans and self.neg_ans[1:-1]!=self.save_file['sentence'] and self.neg_ans!="":
                    break
            except TimeoutError as e:
                flag=1
                if str(e) == "运行超时":
                    print("捕获到运行超时错误，任务被中断。")
                    break
            except Exception as e:
                    print("An error occurred:", e)
                    print("Trying again...")
        if flag==1:
            raise TimeoutError("运行超时")
        self.negative.add_memory(self.neg_ans)

        aff_ans=self.aff_ans
        neg_ans=self.neg_ans
        self.judge.add_event(self.save_file['judge_prompt_last'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
        if aff_ans == neg_ans :
            hh_ans = {"supported side": "Draw","round_1 constituency parsing result": aff_ans} 
            self.judge.add_memory(hh_ans)
            self.save_file.update(hh_ans) #把这个字典更新进去
            self.save_file['success'] = True
        # ultimate deadly technique.
        else:
            cnt=0
            while True:
                try:
                    flag=0
                    cnt+=1
                    if cnt>2:
                        ans = {"supported side": "Negative","round_1 constituency parsing result": neg_ans}
                        break
                    ans = self.judge.ask()
                   # import pdb;pdb.set_trace()
                    ans = eval(fix_missing_quotes(Find_Start_End_with_Label(ans, "big")))
                    ans = find_keyword_values(ans)
                    if ans["supported side"]=='Affirmative':
                        ans = {"supported side": "Affirmative","round_1 constituency parsing result": aff_ans}
                        break
                    elif ans["supported side"]=='Negative':
                        ans = {"supported side": "Negative","round_1 constituency parsing result": neg_ans}
                        break
                except TimeoutError as e:
                    flag=1
                    if str(e) == "运行超时":
                        print("捕获到运行超时错误，任务被中断。")
                        break
                except Exception as e:
                        print("An error occurred:", e)
                        print("Trying again...")
            if flag==1:
                raise TimeoutError("运行超时")
            self.judge.add_memory(ans)
            if ans != '':
                self.save_file['success'] = True
                # save file
            self.save_file.update(ans)


    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]
            
    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players. 
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players. 

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)


    def run(self):

        for round in range(self.max_round - 1):
            print(f"===== Debate Round-{round+2} =====\n")
            #.remove_last_memory()
            #import pdb;pdb.set_trace()
            
            self.affirmative.add_event(self.save_file['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
           
            flag=0
            cnt=0
            while True:
                try:
                    flag=0
                    cnt+=1
                    if cnt>3:
                        self.aff_ans='('+self.save_file['sentence']+')'
                        break
                    self.aff_ans1 = self.affirmative.ask()
                   # print(self.aff_ans1)
                    #import pdb;pdb.set_trace()
                    try:
                        self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('My_result')[-1].split('Your_result')[-1].split("correction:")[-1].split("correct result")[-1]) 
                    except Exception as e:
                        self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[0])
                    if len(self.save_file['sentence'].split())<3:
                        break
                    if self.aff_ans[1:-1] ==self.save_file['sentence']:
                        self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[0])
                    if self.aff_ans[1:-1] ==self.save_file['sentence']:
                        self.aff_ans = Format_output(self.save_file['sentence'], self.aff_ans1.split('\n')[-1])
                    self.aff_ans=self.aff_ans.replace("()","")
                    
                    if "()" not in self.aff_ans and self.aff_ans[1:-1]!=self.save_file['sentence'] and self.aff_ans!="":
                        break
                except TimeoutError as e:
                    flag=1
                    if str(e) == "运行超时":
                        print("捕获到运行超时错误，任务被中断。")
                        break
                except Exception as e:
                        print("An error occurred:", e)
                        print("Trying again...")
            if flag==1:
                raise TimeoutError("运行超时")
            self.affirmative.add_memory(self.aff_ans)

            self.negative.add_event(self.save_file['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
           
            cnt=0
            while True:
                try:
                    flag=0
                    cnt+=1
                    if cnt>3:
                        self.neg_ans='('+self.save_file['sentence']+')'
                        break
                    self.neg_ans1 = self.negative.ask()
                    # print(self.neg_ans1)
                    # import pdb;pdb.set_trace()
                    try:
                        self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('My_result: ')[-1].split('Your_result: ')[-1].split("correction:")[-1].split("correct result")[-1])
                    except Exception as e:
                        self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[0])
                    if len(self.save_file['sentence'].split())<3:
                        break
                    if self.neg_ans[1:-1] ==self.save_file['sentence']:
                        self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[0])
                    if self.neg_ans[1:-1] ==self.save_file['sentence']:
                        self.neg_ans = Format_output(self.save_file['sentence'], self.neg_ans1.split('\n')[-1])
                    self.neg_ans=self.neg_ans.replace("()","")
                    
                    if "()" not in self.neg_ans and self.neg_ans[1:-1]!=self.save_file['sentence'] and self.neg_ans!="":
                        break
                except TimeoutError as e:
                    flag=1
                    if str(e) == "运行超时":
                        print("捕获到运行超时错误，任务被中断。")
                        break
                except torch.cuda.OutOfMemoryError as e:
                    break
                except Exception as e:
                        print("An error occurred:", e)
                        print("Trying again...")
            if flag==1:
                raise TimeoutError("运行超时")
            self.negative.add_memory(self.neg_ans)  

            
            aff_ans = self.affirmative.memory_lst[-1]['content']
            neg_ans = self.negative.memory_lst[-1]['content']
            self.judge.add_event(self.save_file['judge_prompt_last'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            if aff_ans == neg_ans :
                hh_ans = {"supported side": "Draw","round_{} constituency parsing result".format(round+2): aff_ans} 
                self.judge.add_memory(hh_ans)
                self.save_file.update(hh_ans) #把这个字典更新进去
                self.save_file['success'] = True
            # ultimate deadly technique.
            else:
                cnt=0
                while True:
                    try:
                        flag=0
                        cnt+=1
                        if cnt>3:
                            ans = {"round_{} constituency parsing result".format(round+2): aff_ans}
                            break
                        ans = self.judge.ask()
                       # import pdb;pdb.set_trace()
                        ans = eval(fix_missing_quotes(Find_Start_End_with_Label(ans, "big")))
                        ans = find_keyword_values(ans)
                        if ans["supported side"]=='Affirmative':
                            ans = {"supported side": "Affirmative","round_{} constituency parsing result".format(round+2): aff_ans}
                            break
                        elif ans["supported side"]=='Negative':
                            ans = {"supported side": "Negative","round_{} constituency parsing result".format(round+2): neg_ans}
                            break
                    except TimeoutError as e:
                        flag=1
                        if str(e) == "运行超时":
                            print("捕获到运行超时错误，任务被中断。")
                            break
                    except Exception as e:
                            print("An error occurred:", e)
                            print("Trying again...")
                if flag==1:
                    raise TimeoutError("运行超时")
                self.judge.add_memory(ans)
                if ans != '':
                    self.save_file['success'] = True
                    # save file
                self.save_file.update(ans)

        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst


def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-llama_i", "--llama-input-file", type=str,help="llama Input file path")
    parser.add_argument("-vicuna_i", "--vicuna-input-file", type=str, help="vicuna Input file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-m", "--model-name", type=str, default="llama2", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")

    return parser.parse_args()

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
        #tokenizer = Tokenizer(model_path="/public/home/dzhang/pyProject/hytian/ZModel/llama-main/llama-2-7b-chat-hf/tokenizer.model")
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
            ) # fix zwq
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




if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    args = parse_args()
    
    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 2)[0]
    config = json.load(open(f"{MAD_path}/code/utils/config4cp_lora_coco.json", "r"))

    
    #读取videollava模型结果
    cache_dir = 'cache_dir'
    device = torch.device("cuda:0")
    load_4bit, load_8bit = False, False
    videollava_model_path = '/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main/checkpoints/videollava-7b-lora'
    videollava_model_base='/public/home/dzhang/pyProject/hytian/XModel/Video-LLaVA-main/Video-LLaVA-7B'
    model_name = get_model_name_from_path(videollava_model_path)   
    videollava_tokenizer, videollava_model, videollava_processor, _ = load_pretrained_model(videollava_model_path, videollava_model_base, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)

    device = torch.device("cuda:1")
    #读取intern2-vl模型结果
    path = '/public/home/dzhang/pyProject/hytian/XModel/InternVL-main/output/internvl2-8b/v5-20250103-102018/checkpoint-112-merged'
    intern2_vl_model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    intern2_vl_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
 

    save_file_dir = args.output_dir
    if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    llm_pred_spans = []
    parser_pred_spans = []
    gold_spans = [] 
    gold_labels = []
    max_lengths = []
    problem_data = []
    llm_pred_data = []

    # test_imgs_path='/public/home/dzhang/pyProject/hytian/YModel/MSCOCO/images_list/test2014_imgs_path_list.json'
    # test_gold_caps_path='/public/home/dzhang/pyProject/hytian/YModel/MSCOCO/mscoco/test_gold_caps.json'
    
  
    test_gold_caps_path='/public/home/dzhang/pyProject/hytian/XModel/Qwen2-VL/youcookii_text_to_video_last.json'

    
    with open(test_gold_caps_path,'r',encoding='utf-8') as file:
        line_cnt=-1

        for line in file:
            # if line_cnt+1 not in [1049]:
            #     line_cnt+=1
            #     continue
            if line_cnt>=1200:  #100张图,500个条字幕
                break
            # if line_cnt<1200:
            #     line_cnt+=1
            #     continue
            line_cnt+=1

            data = json.loads(line)
            # 获取 id 键对应的值
            id_value = data.get('id')
            
            # 获取另一个键对应的值（假设这个键为 'other_key'）
            caps= data.get("caption")
            spans = data.get("span")
            labels = data.get("label")
            image_file = data.get('video_path')


            
            prompts_path = f"{save_file_dir}/{line_cnt}-config.json"
            config['sentence']=caps
            config['gold_label']=labels
            config['gold_span']=spans
            config['img']=image_file 
            with open(prompts_path, 'w') as file: #创建保存文件0-config.json
                json.dump(config, file, ensure_ascii=False, indent=4)
            #设置超时的处理器
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(450)  # 设置超时时间为600秒
            try:
                debate = Debate(save_file_dir=save_file_dir, num_players=3, prompts_path=prompts_path, temperature=0, sleep_time=0,videollava_model=videollava_model, videollava_tokenizer=videollava_tokenizer,videollava_processor=videollava_processor,intern2_vl_model=intern2_vl_model, intern2_vl_tokenizer=intern2_vl_tokenizer, image_file=config['img'])
                debate.run()
                debate.save_file_to_json(line_cnt)
                
                
            except  Exception as e:
                with open("data/problem.txt", 'a+', encoding='utf-8') as problem_file:
                    temp = str(line_cnt)+"\t\t"+caps+"\n"
                    problem_file.write(temp)
                    problem_data.append(temp)
                    continue
            # 取消警报
            signal.alarm(0)
            

