import numpy as np
with open(r'E:\WorkSpace\PycharmProject\NLP\Tencent_AILab_ChineseEmbedding\Tencent_AILab_ChineseEmbedding.txt','r',encoding='utf-8') as f:
    f.readline()#第一行为词汇数和向量维度，在这里不予展示
    f.readline()
    m=f.readline()#读取第三个词
    vecdic = dict()#构造字典
    vectorlist = m.split()#切分一行，分为词汇和词向量
    vector = list(map(lambda x:float(x),vectorlist[1:]))#对词向量进行处理
    vec = np.array(vector)#将列表转化为array
    vecdic[vectorlist[0]]=vec
    print(vectorlist[0])
    print(vecdic['的'])
