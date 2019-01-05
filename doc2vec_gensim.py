# -*- coding: utf-8 -*-
import sys
import logging
import os
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import gensim
from gensim.models import Doc2Vec

def get_trainset(path):
    """
    加载预料，将其整理成规定的形式，使用到TaggedDocument模型
    :param keyword_path:
    :return:返回doc2vec规定的数据格式
    """
    documents = []
    # 使用count当做每个句子的“标签”，标签和每个句子是一一对应的
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        docs = f.readlines()
        for doc in docs:
            words = doc.strip('\n')
            # 这里documents里的每个元素是二元组，具体可以查看函数文档
            documents.append(gensim.models.doc2vec.TaggedDocument(words, [str(count)]))
            count += 1
            if count % 10000 == 0:
                logging.info('{} has loaded...'.format(count))
    return documents

def train_docVecs(documents_train, size=200, epoch_num=1):
    """
    训练文本，得到文本向量
    :param documents_train: 训练数据，必须要TaggedDocument格式
    :param size: 表示生成的向量纬度
    :param window: 表示训练的窗口大小，也就是当前词与预测词在一个句子中的最大距离是多少
    :param min_count:表示参与训练的最小词频,也就是对字典做截断，词频少于min_count次数的单词会被丢弃掉，默认值为5
    :param dm:表示训练的算法，默认为1。dm=0时，则使用DBOW
    :param alpha:分为start_alpha和end_alpha两个参数，表示初始的学习速率，在训练过程中会线性地递减到min_alpha
    :param max_vocab_size：设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。
                            每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    :param sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，官网给的解释 1e-5效果比较好。设置为0时是词最少的时候！
    :param workers：用于控制训练的并行数。
    :param total_examples: Count of sentences.
    :param total_words : Count of raw words in documents.
    :param negtive:
    :param epochs:迭代次数，默认继承了word2vec的5次
    :return: 返回的是训练好的模型。
    """
    model = Doc2Vec(documents_train, min_count=1, window=3, size=size,
                    sample=1e-3, negative=5, workers=4, epochs= epoch_num)
    model.train(documents_train, total_examples=model.corpus_count, epochs=300)
    return model

def save_docVecs(model, doc2vec_path):
    # 保存模型
    model.save(doc2vec_path)

def test_doc2vec(load_path):
    # 加载模型
    model = Doc2Vec.load(load_path)
    # 输出标签为‘10’句子的向量
    print(model.docvecs['10'])

def cluster(x_train,load_path, cluster_path2, kmean_model_path):
    """
    k均值聚类
    :param x_train: 数据
    :param load_path: doc2vec模型路径
    :param cluster_path: 聚类结果输出的存放路径
    :param kmean_model_path: K均值聚类模型的存放路径
    :return: 返回类簇中心向量
    """
    infered_vectors_list= []
    print("load doc2vec model...")
    model_dm = Doc2Vec.load(load_path)
    print("load train vectors...")
    i = 0
    for text, label in x_train:
        #得到每个文本对应的文档向量
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    print("train kmean model...")
    # 设定簇数
    kmean_model = KMeans(n_clusters=85)
    # 模型训练
    kmean_model.fit(infered_vectors_list)

    # 簇的中心向量
    cluster_centers = kmean_model.cluster_centers_
    # 保存模型，载入模型使用model = joblib.load(kmean_model_path)
    joblib.dump(kmean_model,kmean_model_path)
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好。选取临界点的簇个数
    #print(kmean_model.inertia_)

    """
    #每个样本所属的类簇,并将训练结果输出
    Labels = kmean_model.labels_
    with open(cluster_path1, 'w') as wf1:
        for i in range(4861):
            string = ""
            text = x_train[i][0]
            for word in text:
                string = string + word
            string = '该文本属于类簇'+ str(Labels[i]) + ':\t' + string
            string = string + '\n'
            wf1.write(string)
    """
    # 通过训练的模型预测前1000个文本所属类簇标签
    labels = kmean_model.predict(infered_vectors_list[0:4851])
    with open(cluster_path2, 'w') as wf2:
        for i in range(4851):
            string = ""
            text = x_train[i][0]
            for word in text:
                string = string + word
            #string = string + '\t'
            string = '该文本属于类簇'+ str(labels[i]) + ':\t' + string
            string = string + '\n'
            wf2.write(string)

    return cluster_centers

if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    # 引入日志配置
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #加载数据
    keyword_path = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\keword_result\\keyword_result.txt'
    source_path = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\source\\Teachr_Data_delNum.txt'
    segment_path = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\segmen_result\\seg_result.txt'
    #docus = get_trainset(keyword_path)
    #docus = get_trainset(segment_path)
    docus = get_trainset(source_path)

    #训练数据
    #doc_model= train_docVecs(docus)

    #训练并保存数据
    docs2vec_path = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\models\\segment_d2v.model'
    doc_model = train_docVecs(docus)
    save_docVecs(doc_model, docs2vec_path)

    #测试数据
    # test_doc2vec(docs2vec_path)
    #K均值聚类
    #cluster_path1 = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\cluster_result\\claffify_train.txt'
    cluster_path2 = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\cluster_result\\claffify_segment.txt'
    kmean_model_path = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\models\\segment_kmeans.model'
    cluster_centers = cluster(docus, docs2vec_path, cluster_path2, kmean_model_path)





