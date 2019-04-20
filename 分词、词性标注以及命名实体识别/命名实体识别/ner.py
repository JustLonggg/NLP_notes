#encoding=utf8
import jieba
import re
from rules import grammer_parse

with open('text.txt','r',encoding='utf-8') as fp:
    with open('out.txt','w',encoding='utf-8') as fout:
        [grammer_parse(line.strip(),fout) for line in fp if len(line.strip()) > 0]
        