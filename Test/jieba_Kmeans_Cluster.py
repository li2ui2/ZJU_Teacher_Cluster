#!/usr/bin/env python
# coding=utf-8
import sys, os
import numpy as np
from numpy import *
import jieba
import math
import jieba.analyse

#载入用户自定义的吃点，以便包含jieba词库里没有的词
#词典格式：一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒
jieba.load_userdict("E:/WorkSpace/PycharmProject/Crawling_NLP/Data/userdict.txt")

def read_from_file(file_name):
    """
    读取语料中的内容，并存放于words中
    :param file_name: 语料文件的路径
    :return: words
    """
    with open(file_name) as fp:
        words = fp.read()
    return words

def stop_words(stop_word_file):
    """
    #对停用词进行，并存放于new_words列表中
    :param stop_word_file:
    #set() 函数创建一个无序不重复元素集合{}，可进行关系测试，删除重复数据，还可以计算交集&、差集-、并集|等。
    :return: set(new_words)
    """
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)

def del_stop_words(words, stop_words_set):
    """
    :param words: 未分词的文档
    :param stop_words_set: 停用词文档
    :return: 去除停用词后的文档
    """
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
            # print r.encode("utf-8"),
    # print len(new_words),len(set(new_words))
    return new_words

def tfidf(term, doc, word_dict, docset):
    tf = float(doc.count(term)) / (len(doc) + 0.001)
    idf = math.log(float(len(docset)) / word_dict[term])
    return tf * idf

def idf(term, word_dict, docset):
    idf = math.log(float(len(docset)) / word_dict[term])
    return idf

def word_in_docs(word_set, docs):
    word_dict = {}
    for word in word_set:
        # print word.encode("utf-8")
        word_dict[word] = len([doc for doc in docs if word in doc])
        # print word_dict[word],
    return word_dict

#构建词袋空间VSM(vector space model)
def get_all_vector(file_path, stop_words_set):
    """
    最终得到的矩阵的性质为：
    列是所有文档总共的词的集合；每行代表一个文档；每行是一个向量，向量的每个值是这个词的权值。
    :param file_path:
    :param stop_words_set:
    :return:names：文本路径
             tfidf：权值
    """
    #os.path.join(path1[, path2[, ...]])  把目录和文件名合成一个路径
    #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。

    # 1.将所有文档读入到程序中
    #names为构建的语料库中每个txt文件对应的路径列表
    names = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    posts = [open(name, encoding='utf-8').read() for name in names]

    # 2.对每个文档进行切词，并去除文档中的停用词。 3. 统计所有文档的词集合
    docs = []
    word_set = set()
    #将所有文档分词后的结果都存放于word_set集合当中，并将一个文档作为一个列表项存放于docs列表中
    for post in posts:
        doc = del_stop_words(post, stop_words_set)
        docs.append(doc)
        word_set |= set(doc)
        # print len(doc),len(word_set)

    # 4. 对每个文档，都将构建一个向量，向量的值是词语在本文档中出现的次数。
    word_set = list(word_set)  #转换为列表
    docs_vsm = []
    # for word in word_set[:30]:
    # print word.encode("utf-8"),
    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0) #统计词在文档中出现的次数
        # print temp_vector[-30:-1]
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)  #将词频列表转换为矩阵
    # print docs_matrix.shape
    # print len(np.nonzero(docs_matrix[:,3])[0])

    # 5.将单词出现的次数转换为权值(TF-IDF）
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    idf = np.log(column_sum)
    idf = np.diag(idf)
    # print idf.shape
    # row_sum    = [ docs_matrix[i].sum() for i in range(docs_matrix.shape[0]) ]
    # print idf
    # print column_sum
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())

    tfidf = np.dot(docs_matrix, idf)

    return names, tfidf

def gen_sim(A, B):
    """
    文本相似度计算，该函数计算余弦相似度
    余弦相似度用向量空间中两个向量夹角的余弦值作为衡量两个个体差异的大小。
    相比欧氏距离度量，余弦相似度更加注重两个向量在方向上的差异，而非距离或长度上的差异。
    相对于欧氏距离，余弦相似度更适合计算文本的相似度。首先将文本转换为权值向量，
    通过计算两个向量的夹角余弦值，就可以评估他们的相似度。
    余弦值的范围在[-1,1]之间，值越趋近于1，代表两个向量方向越接近；越趋近于-1，代表他们的方向越相反。
    为了方便聚类分析，我们将余弦值做归一化处理，将其转换到[0,1]之间，并且值越小距离越近。
    :param A:
    :param B:
    :return:
    """
    num = float(np.dot(A, B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn   # 余弦值为[-1,1],归一化为[0,1],值越大相似度越大
    return sim

def randCent(dataSet, k):
    """
    该函数为给定数据集构建一个包含k个随机初始聚类中心的集合
    :param dataSet:数据集合，矩阵
    :param k: 初始聚类中心个数
    :return:
    """
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    """
    该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    这个过程重复数次，直到数据点的簇分配结果不再改变为止。

    该函数接受4个参数，只有数据集及簇的数目是必选的参数，用来计算距离和创建初始质心的函数都是可选的。
    :param dataSet:数据集合，矩阵
    :param k:初始聚类中心个数
    :param distMeas:  计算距离的函数
    :param createCent: 创建初始质心的函数
    :return:clusterAssment：簇分配结果矩阵，包含两列：一列记录簇索引值，
                             第二列存储误差(这里的误差是指当前点到簇质心的距离，可用该误差来评价聚类效果)
             centroids：簇中心的结果
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf;
            minIndex = -1
            #寻找最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        #更新质心的位置
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment

if __name__ == "__main__":
    stop_words = stop_words("E:/WorkSpace/PycharmProject/Crawling_NLP/Data/stopwords.txt")
    names, tfidf_mat = get_all_vector("E:/WorkSpace/PycharmProject/Crawling_NLP/Data/Teacher_Data2/", stop_words)
    print(tfidf_mat)
    # file = open("E:/WorkSpace/PycharmProject/Crawling_NLP/Data/result.txt" , 'w', encoding='utf-8')
    # file.write(tfidf_mat)
    # file.close()
    np.savetxt("E:/WorkSpace/PycharmProject/Crawling_NLP/Data/result.txt", tfidf_mat)
    myCentroids, clustAssing = kMeans(tfidf_mat, 5, gen_sim, randCent)
    for label, name in zip(clustAssing[:, 0], names):
        print(label, name)

