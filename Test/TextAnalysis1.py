# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 15:18:29 2016

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
from scipy import ndimage
from sklearn import manifold, datasets
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

####第一步 计算TFIDF####

# 文档预料 空格连接
corpus = []

# 读取预料 一行预料为一个文档
for line in open('E:/WorkSpace/PycharmProject/Text_cluster/Data/Teacher_Data.txt', 'r',encoding='utf-8').readlines():
    # print line
    corpus.append(line.strip())
# print corpus
# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()

# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()

# 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()

# 打印特征向量文本内容
#print('Features length:' + str(len(word)))
resName = "BHTfidf_Result.txt"
result = codecs.open(resName, 'w', 'utf-8')
for j in range(len(word)):
    result.write(word[j] + ' ')
result.write('\r\n\r\n')

# 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weight)):
    # print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
    for j in range(len(word)):
        # print weight[i][j],
        result.write(str(weight[i][j]) + ' ')
    result.write('\r\n\r\n')

result.close()

####第二步 聚类Kmeans####
print('Start Kmeans:')
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=50)  # 景区 动物 人物 国家
s = clf.fit(weight)
print(s)

# 中心点
print(clf.cluster_centers_)

# 每个样本所属的簇
# label = []  # 存储1000个类标 4个类
# print(clf.labels_)
# i = 1
# while i <= len(clf.labels_):
#     print(i, clf.labels_[i - 1])
#     label.append(clf.labels_[i - 1])
#     i = i + 1

# 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  958.137281791
print(clf.inertia_)

# ####第三步 图形输出 降维####
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)  # 输出两维
# newData = pca.fit_transform(weight)  # 载入N维
# print(newData)
#
# # 5A景区
# x1 = []
# y1 = []
# i = 0
# while i < 400:
#     x1.append(newData[i][0])
#     y1.append(newData[i][1])
#     i += 1
#
# # 动物
# x2 = []
# y2 = []
# i = 400
# while i < 600:
#     x2.append(newData[i][0])
#     y2.append(newData[i][1])
#     i += 1
#
# # 人物
# x3 = []
# y3 = []
# i = 600
# while i < 800:
#     x3.append(newData[i][0])
#     y3.append(newData[i][1])
#     i += 1
#
# # 国家
# x4 = []
# y4 = []
# i = 800
# while i < 1000:
#     x4.append(newData[i][0])
#     y4.append(newData[i][1])
#     i += 1
#
# # 四种颜色 红 绿 蓝 黑
# plt.plot(x1, y1, 'or')
# plt.plot(x2, y2, 'og')
# plt.plot(x3, y3, 'ob')
# plt.plot(x4, y4, 'ok')
# plt.show()