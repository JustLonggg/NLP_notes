#encoding=utf8
import json
import sys,os,re
import numpy
from tokenizer import seg_sentences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pattern = re.compile(u'[^a-zA-Z\u4E00-\u9FA5]')


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.float):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()        
        return json.JSONEncoder.default(self, obj)


def _replace_c(text):
    intab = ',?!'
    outtab = '，？！'
    deltab = ')(+_-.<> '
    trantab = text.maketrans(intab,outtab,deltab)
    return text.translate(trantab)


# 先以标点符号为单位切分，再使用hanlp的seg_sentences分词
def tokenize_raw(text):
    split_sen = (i.strip() for i in re.split(u'。|,|，|:|：|?|!|\t|\n',_replace_c(text)) if len(i.strip())>4)
    # 使用（）生成器来减小内存
    return [seg_sentences(sentence) for sentence in split_sen]


def list_2_ngram(sentence,n=4,m=2):
    if len(sentence) < n:
        n = len(sentence)
    temp = [tuple(sentence[i-k:i]) for k in range(m,n+1) for i in range(k,len(sentence) + 1)]
    return [item for item in temp if len(''.join(item).strip())>1 and len(pattern.findall(''.join(item).strip()))==0]


if __name__ == '__main__':
    copus = [tokenize_raw(line.strip()) for line in open('text.txt','r',encoding='utf-8') if len(line.strip())>0 and "RESUMEDOCSSTARTFLAG" not in line]
    doc = []
    if len(copus) > 1:
        for list_copus in copus:
            for t in list_copus:
                doc.extend([' '.join(['_'.join(i) for i in list_2_ngram(t,n=4,m=2)])])
    doc = list(filter(None,doc))
    fout = open('ngram2_4.txt','w',encoding='utf-8')

    # 使用tfidf计算频率
    vectorizer1 = CountVectorizer()   # 初始化一个计数类

    transformer = TfidfTransformer()    # 该类会统计每个词语的tf-idf权值
    freq1 = vectorizer1.fit_transform(doc)
    tfidf = transformer.fit_transform(freq1)

    word_freq = [freq1.getcol(i).sum() for i in range(freq1.shape[1])]
    tfidf_sum = [tfidf.getcol(i).sum() for i in range(tfidf.shape[1])]

    tfidf_dic = vectorizer1.vocabulary_
    tfidf_dic = dict(zip(tfidf_dic.values(),tfidf_dic.keys()))  # 反转

    dic_filter = {}
    def _add(wq,tf,i):
        dic_filter[tfidf_dic[i]] = [wq,tf]
    for i,(word_freq_one,w_one) in enumerate(zip(word_freq,tfidf_sum)):
        _add(word_freq_one,w_one,i)
    sort_dic = dict(sorted(dic_filter.items(),key=lambda val:val[1],reverse=True))
    fout.write(json.dumps(sort_dic, ensure_ascii=False,cls=NumpyEncoder))               
    fout.close()