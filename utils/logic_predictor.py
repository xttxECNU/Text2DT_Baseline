def logic_pre(triples,text):
    index=[]
    and_words = ['且', '同时', '再', '然后', '联合', '继而', '并', '还需', '+', '与', '的']
    or_words = ['或']
    dict={"临床表现":0,"治疗药物":0,"治疗方案":0,"用法用量":0,"基本情况":0,"禁用药物":0}
    for triple in triples:
        dict[triple[1]] +=1
    if len(triples) < 2:
        return 'null'
    else:
        if dict["禁用药物"]>0:
            return 'and'
        elif dict["治疗方案"]+dict["治疗药物"]>1 or dict["基本情况"]>1 or dict["临床表现"] > 1 or dict["基本情况"]+dict["临床表现"] > 1:
            for triple in triples:
                index.append(text.index(triple[2]))
            for word in or_words:
                if word in text[min(index):max(index)]:
                    return "or"
            for word in and_words:
                if word in text[min(index):max(index)]:
                    return "and"
        elif dict["用法用量"]>0:
            return 'and'
    return "or"

