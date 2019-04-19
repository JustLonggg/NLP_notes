说明

分别通过jieba和hanlp加载自定义词典来进行中文分词
jieba只需通过jieba.load_userdict("dict.txt")即可
而hanlp需要修改配置文件
1.在D:/hanlp/data/dictionary/custom/"下添加字典（格式为 “名称 词性 词频”，注意修改为utf-8）

2.删除"CustomDictionary.txt.bin"

3.在D:\hanlp\hanlp.properties中添加字典

最后运行 cut_data.py
