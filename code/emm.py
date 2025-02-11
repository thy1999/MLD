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


import signal
import time

# 定义处理超时的函数
def handler(signum, frame):
    raise TimeoutError("运行超时")

NAME_LIST=[
    "Affirmative side",
    "Negative side",
    "Judge_round2",
    "Judge_round3",
]


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

 ### 根据标签找到对应的文本并进行提取。
def Find_Start_End_with_Label(ori_str, label):
    if label == "mid":
        return ori_str[ori_str.find("["):ori_str.rfind("]")+1]
    elif label == "big":
        return ori_str[ori_str.find("{"):ori_str.find("}")+1]

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
    count = 0
    flag = 1
    for char in text:
        if stack==[] and count!=0:
            flag=0
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                # Ideally, this situation should not occur if input is expected to have more '[' than ']'
                text = '(' + text  # Prepend a missing '['
        count+=1
    # Append missing ']' for each unmatched '['
    text += ')' * len(stack)
    if flag==0:
        text = "("+text+")"
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

# def match_words(sentence, pred):
#     ### 单词匹配，保证原始句子中的单词与嵌套列表中的单词一致。
#     nested_lst = small_to_mid_transform(pred)
#     assert type(nested_lst)==list
#     nested_str = str(nested_lst)
#     words_list = sentence.split()
#     st = 0
#     new_s = ""
#     i = 0
#     while i<len(nested_str):
#         tp = nested_str[i]
#         if tp=="'" or tp=='"':
#             j = i+1
#             if tp == "'":
#                 while (j<len(nested_str) and nested_str[j]!="'"):
#                     j+=1
#             elif tp == '"':
#                 while (j<len(nested_str) and nested_str[j]!='"'):
#                     j+=1
#             nest_word = nested_str[i+1:j]
#             if st < len(words_list):
#                 if nest_word == words_list[st]:
#                     new_s+=nested_str[i:j+1]
#                     st+=1
#                     i=j+1
#                 else:
#                     last = st
#                     st += 1 
#                     while st<len(words_list) and nest_word!=words_list[st]:
#                         st += 1
#                     if st>=len(words_list):
#                         st = last
#                         if j+1<len(nested_str):
#                             if nested_str[j+1]==",":
#                                 i = j+2
#                             elif nested_str[j+1]=="]":
#                                 # if nested_str[i-1]==" ":
#                                 #     new_s = new_s[:-2]
#                                 i = j+1
#                     else:
                        
#                         for k in range(last,st):
#                             new_s+=("'"+str(words_list[k]).replace("'","\\'")+"', ")
#                         new_s+=nested_str[i:j+1]
#                         st+=1
#                         i=j+1
#             else:
                
#                     i = j+1
#         else:
            
#             if st>=len(words_list) and tp!=']':
#                 new_s=new_s.strip(",")
#                 count_lf=new_s.count("[")
#                 count_rf=new_s.count("]")
#                 if count_lf>count_rf:
#                     new_s=new_s + (count_lf- count_rf)*']'
#                 break
#             new_s += tp
#             i += 1
#     if st!=len(words_list):
#         new_s = str(eval(new_s))
#         ind = new_s.rfind("'")
#         temp_list = ["'"+k+"'" for k in words_list[st:len(words_list)]]
#         temp = ", ".join(temp_list)
#         new_s = new_s[:ind+1]+", "+temp+new_s[ind+1:]
#     new_s = eval(new_s.replace('[]', ''))
#     assert extract_words(new_s)==words_list
    
#     return to_parentheses_format(new_s)

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
    #     import pdb;pdb.set_trace()
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




#这个标点符号
aff_ans1='((coolermaster V8) (] -LRB- ((the man) (said ((the backplate) (is (probably (shorting ((the pointy bits) (on ((the back) (of (the mobo)))))))), (so (he (had (me (loosen (the heatsink) (some)), and ((whaddyah know), (it (turned on and (worked (just fine!))))))))))))).)'
sentence='coolermaster V8 ] -LRB- the man said the backplate is probably shorting the pointy bits on the back of the mobo , so he had me loosen the heatsink some , and whaddyah know , it turned on and worked just fine !'


import pdb;pdb.set_trace()
aff_ans = Format_output(sentence, aff_ans1.split('My_result: ')[-1].split('Your_result: ')[-1].split("correction:")[-1].split("correct result")[-1])
import pdb;pdb.set_trace()
print(aff_ans)