#encoding=utf8

dict_file = open(r'D:\hanlp\data\dictionary\custom\resume_nouns.txt','r',encoding='utf-8')
d = {}

[d.update({line:len(line.split(' ')[0])}) for line in dict_file]    # dict_file == dict_file.readlines()
f = sorted(d.items(),key=lambda x:x[1],reverse=True)
new_file = open(r'D:\hanlp\data\dictionary\custom\resume_nouns_1.txt','r',encoding='utf-8')
[new_file.write(item[0]) for item in f]
new_file.close()