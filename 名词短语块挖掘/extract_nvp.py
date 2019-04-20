#encoding=utf8
import os,json,nltk,re
from tokenizer import cut_hanlp



huanhang=set(['。','？','！','?'])
keep_pos="q,qg,qt,qv,s,t,tg,g,gb,gbc,gc,gg,gm,gp,mg,Mg,n,an,ude1,nr,ns,nt,nz,nb,nba,nbc,nbp,nf,ng,nh,nhd,o,nz,nx,ntu,nts,nto,nth,ntch,ntcf,ntcb,ntc,nt,nsf,ns,nrj,nrf,nr2,nr1,nr,nnt,nnd,nn,nmc,nm,nl,nit,nis,nic,ni,nhm,nhd"
keep_pos_nouns=set(keep_pos.split(","))
keep_pos_v="v,vd,vg,vf,vl,vshi,vyou,vx,vi,vn"
keep_pos_v=set(keep_pos_v.split(","))
keep_pos_p=set(['p','pbei','pba'])
merge_pos=keep_pos_p|keep_pos_v
keep_flag=set(['：','，','？','。','！','；','、','-','.','!',',',':',';','?','(',')','（','）','<','>','《','》'])
drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])


# 使用for循环遍历树
def getNodes(parent,model_tagged_file):
    text = ''
    for node in parent:
        if type(node) is nltk.Tree:
            # 如果是NP或者VP的话合并分词
            if node.label() == 'NP':
                text += ''.join([node_child[0].strip() for node_child in node.leaves()])+'/NP'+3*' '
            elif node.label() == 'VP':
                text += ''.join([node_chile[0].strip() for node_chile in node.leaves()])+'/VP'+3*' '
        # 如果不是树，就是叶子节点，直接标记词为其他‘O’或者PP
        else:
            if node[1] in keep_pos_p:
                text += node[0].strip()+'/PP'+3*' '
            if node[0] in huanhang:
                text += node[0].strip()+'/O'+3*' '
            if node[1] not in merge_pos:
                text += node[0].strip()+'/O'+3*' '
    model_tagged_file.write(text+'\n')


def grammer(sentence,model_tagged_file):
    # 输入的sentence的格式为：[('工作','vn'),('描述','v'),(':','w')]
    grammer1 = r"""NP:
        {<m|mg|Mg|mq|q|qg|qt|qv|s|>*<a|an|ag>*<s|g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|o|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<f>?<ude1>?<g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|o|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<cc>+<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<m|mg|Mg|mq|q|qg|qt|qv|s|>*<q|qg|qt|qv>*<f|b>*<vi|v|vn|vg|vd>+<ude1>+<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<vi>?}
        VP:{<v|vd|vg|vf|vl|vshi|vyou|vx|vi|vn>+}
        """      # 动词短语块

    cp = nltk.RegexpParser(grammer1)
    try:
        result = cp.parse(sentence)      # 输出以grammer1设置的名词块为单位的树
    except:
        pass
    else:
        getNodes(result,model_tagged_file)   # 使用getNodes遍历树


if __name__ == '__main__':
    with open('nvp.txt','w',encoding='utf-8') as fout:
        with open('text.txt','r',encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                grammer(cut_hanlp(line),fout)
                