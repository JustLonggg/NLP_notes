#encoding=utf8
import jieba
import re
from tokenizer import cut_hanlp

#jieba.load_userdict('dict.txt')

fp = open('dict.txt','r',encoding='utf-8')
# for line in fp:
#     line = line.strip()
#     jieba.suggest_freq(line,tune=True)

[jieba.suggest_freq(line.strip(),tune=True) for line in fp]

if __name__ == '__main__':
    string = '台中正确台中应该不会被切开'
    # 当改变词频时，需要关闭HMM
    words_jieba = ' '.join(jieba.cut(string,HMM=False))   
    print(words_jieba)