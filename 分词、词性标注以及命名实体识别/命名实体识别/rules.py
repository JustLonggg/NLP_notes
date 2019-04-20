#encoding=utf8
import nltk,json
from tool import ner_stanford,cut_stanford

def get_stanfrod_ner_nodes(parent):
    date = ''
    org = ''
    loc = ''
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'DATE':
                date = date+' '+''.join([i[0] for i in node])
            elif node.label() == 'ORGANIZATIONL':
                org = org+' '+''.join([i[0] for i in node])
            elif node.label() == 'LOCATION':
                loc = loc+' '+''.join([i[0] for i in node])

    if len(date) > 0 or len(org) > 0 or len(loc) > 0:
        return {'date':date,'org':org,'loc':loc}
    else:
        return {}

def grammer_parse(raw_sentence=None,file_object=None):
    if len(raw_sentence.strip()) < 2:
        return False
    grammer_dict = {
        'stanford_ner_drop':r"""
        DATE:{<DATE>+<MISC>?<DATE>*}
        {<DATE>+}
        {<TIME>+}
        ORGANIZATIONL:{<ORGANIZATION>+}
        LOCATION:{<LOCATION|STATE_OR_PROVINCE|CITY|COUNTRY>+}
        """
    }

    stanford_ner_drop_rp = nltk.RegexpParser(grammer_dict['stanford_ner_drop'])
    try:
        stanford_ner_drop_result = stanford_ner_drop_rp.parse(ner_stanford(raw_sentence))
        # 通过stanfordnlp的ner之后，再通过nltk的parse进行构建语法树
    except:
        print('the error sentence is {}'.format(raw_sentence))
    # 如果try里面的语句没有问题，就执行else语句
    else:
        stanford_keep_drop_dict = get_stanfrod_ner_nodes(stanford_ner_drop_result)
        if len(stanford_keep_drop_dict) > 0:
            file_object.write(json.dumps(stanford_keep_drop_dict,skipkeys=False,
            ensure_ascii=False,
            check_circular=True,
            allow_nan=True,
            cls=None,
            indent=4,
            separators=None,
            default=None,
            sort_keys=False
            ))