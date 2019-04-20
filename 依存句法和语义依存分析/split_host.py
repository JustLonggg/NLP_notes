#encoding=utf8
from stanfordcorenlp import StanfordCoreNLP 
from nltk import Tree,ProbabilisticTree
import nltk,re
import nltk.tree as tree 

nlp = StanfordCoreNLP(r'D:\stanfordnlp',lang='zh')

def _replace_c(text):
    # 将英文标点符号替换成中文标点符号，并去除html语言的一些标志等噪音
    intab = ',?!()'
    outtab = '，？！（）'
    deltab = '\n<li>< li>+_-.><li \U0010fc01 _'
    trantab = text.maketrans(intab,outtab,deltab)
    return text.translate(trantab)

def parse_sentence(text):
    text = _replace_c(text)    # 文本去噪
    try:
        if len(text.strip())>6:
            return Tree.fromstring(nlp.parse(text.strip()))
            # nlp.parse(text)是将句子变成依存句法树，Tree.fromstring是将str类型的树转换成nltk的结构的树
    except:
        pass


def get_noun_chunk(tree):
    if tree.label() == 'NP':
        nouns_phase = ''.join(tree.leaves())
    #    noun_chunk.append(nouns_phase)
    return nouns_phase




def get_vv_loss_np(tree):
    if not isinstance(tree,nltk.tree.Tree):
        return False
    stack = []
    np = []
    stack.append(tree)
    current_tree = ''
    while stack:
        current_tree = stack.pop() 
        if isinstance(current_tree,nltk.tree.Tree) and current_tree.label()=='VP':
            continue
        elif isinstance(current_tree,nltk.tree.Tree) and current_tree.label()!='NP':
            for i in range(len(current_tree)):
                stack.append(current_tree[i])
        elif isinstance(current_tree,nltk.tree.Tree) and current_tree.label()=='NP':
            np.append(get_noun_chunk(current_tree))
    if np:
        return ''.join(np)
    else:
        return False


def search(tree_in):
    if not isinstance(tree_in,nltk.tree.Tree):
        return False
    vp_pair = []
    stack = []
    stack.append(tree_in)
    current_tree = ''
    while stack:
        tree = stack.pop()
        if isinstance(tree,nltk.tree.Tree) and tree.label()=='ROOT':
            for i in range(len(tree)):
                stack.append(tree[i])
        if isinstance(tree,nltk.tree.Tree) and tree.label()=='IP':    # 简单从句
            for i in range(len(tree)):
                stack.append(tree[i])
        if isinstance(tree,nltk.tree.Tree) and tree.label()=='VP':    # 动词短语
            duplicate = []
            if len(tree) >= 2:
                for i in range(1,len(tree)):
                    if tree[0].label()=='VV' and tree[i].label()=='NP':   # 动词 和 名词短语
                        verb = ''.join(tree[0].leaves())    # 合并动词 leaves是分词
                        noun = get_noun_chunk(tree[i])
                        if verb and noun:
                            vp_pair.append((verb,noun))     # 返回动名词短语对
                            duplicate.append(noun)
                    elif tree[0].label()=='VV' and tree[i].label()!='NP':
                        noun = get_vv_loss_np(tree)
                        verb = ''.join(tree[0].leaves())
                        if verb and noun and noun not in duplicate:
                            duplicate.append(noun)
                            vp_pair.append((verb,noun))
    if vp_pair:
        return vp_pair
    else:
        return False


if __name__ == '__main__':
    with open('dependency.txt','w',encoding='utf-8') as fout:
        with open('text.txt','r',encoding='utf-8') as fp:
            for it in fp:
                s = parse_sentence(it)    # 通过stanfordnlp依存句法分析得到一个句法树，再用nltk包装成树的结构
                res = search(s)
                if res:
                    [fout.write(i[0]+ ' '+i[1]+'\n') for i in res]
    print('finish...')