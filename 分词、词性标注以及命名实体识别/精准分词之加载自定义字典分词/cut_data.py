#encoding=utf8
import jieba
import re
from tokenizer import cut_hanlp

jieba.load_userdict('dict.txt')
# hanlp添加自定义字典需要修改配置文件

def merge_two_list(a,b):
    c = []
    len_a,len_b = len(a),len(b)
    for i in range(len_b):
        c.append(a[i])
        c.append(b[i])

    c.append(a[len_a-1])
    return c  

if __name__ == '__main__':
    fp = open('text.txt','r',encoding='utf-8')
    fout_jieba = open('result_cut_jieba.txt','w',encoding='utf-8')
    fout_hanlp = open('result_cut_hanlp.txt','w',encoding='utf-8')
    regex1 = u'[^\u4e00-\u9fa5（）*&……%￥$，,。.@! ！]{1,5}期'  #非汉字和特殊字符的xxx期
    regex2 = r'[0-9]{1,3}[.]?[0-9]{1,3}%'
    p1 = re.compile(regex1)
    p2 = re.compile(regex2)
    for line in fp.readlines():
        result1 = p1.findall(line)
        if result1:
            line = p1.sub('Flag1',line)
        result2 = p2.findall(line)
        if result2:
            line = p2.sub('Flag2',line)

        words = jieba.cut(line)
        result = ' '.join(words)

        words1 = cut_hanlp(line)
        if 'Flag1' in result:
            result = result.split('Flag1')
            result = merge_two_list(result,result1)
            result = ''.join(result)
        if 'Flag2' in result:
            result = result.split('Flag2')
            result = merge_two_list(result,result2)
            result = ''.join(result)
        if 'Flag1' in words1:
            words1 = words1.split('Flag1')
            words1 = merge_two_list(words1,result1)
            words1 = ''.join(words1)
        if 'Flag2' in words1:
            words1 = words1.split('Flag2')
            words1 = merge_two_list(words1,result2)
            words1 = ''.join(words1)
        fout_jieba.write(result)
        fout_hanlp.write(words1)
    fout_jieba.close()
    fout_hanlp.close()
    
