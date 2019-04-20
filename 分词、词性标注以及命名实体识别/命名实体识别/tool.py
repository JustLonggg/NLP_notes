#encoding=utf8
import os,gc,re,sys
from itertools import chain
from stanfordcorenlp import StanfordCoreNLP 

stanford_nlp = StanfordCoreNLP(r'D:\stanfordnlp',lang='zh')

drop_pos_set = set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])
han_pattern = re.compile(r'[^\dA-Za-z\u3007\u4E00-\u9FCB\uE815-\uE864]+')

def ner_stanford(raw_sentence,return_list=True):
    if len(raw_sentence.strip()) > 0:
        return stanford_nlp.ner(raw_sentence) if return_list else iter(stanford_nlp.ner(raw_sentence))

def cut_stanford(raw_sentence,return_list=True):
    if len(raw_sentence.strip()) > 0:
        return stanford_nlp.pos_tag(raw_sentence) if return_list else iter(stanford_nlp.pos_tag(raw_sentence))