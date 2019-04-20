#encoding=utf8
import os,gc,re,sys
from jpype import *

startJVM(getDefaultJVMPath(),'-Djava.class.path=D:\hanlp\hanlp-1.7.2.jar;D:\hanlp',
    '-Xms1g','-Xmx1g')

Tokenizer = JClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')

def to_string(sentence,return_generator=False):
    # 返回生成器
    if return_generator:
        return (word_pos_item.toString().split('/') for word_pos_item in Tokenizer.segment(sentence))
    # 返回字符串
    else:
        return [(word_pos_item.toString().split('/')[0],word_pos_item.toString().split('/')[1]) for word_pos_item in Tokenizer.segment(sentence)]


def cut_hanlp(raw_sentence,return_list=True):
    if len(raw_sentence.strip())>0:
        return to_string(raw_sentence) if return_list else iter(to_string(raw_sentence))