import jieba.analyse
import  codecs
file = codecs.open("E:/WorkSpace/PycharmProject/Text_cluster/Data/Teacher_Data/宋广华.txt",'rb',encoding='utf-8').read()
# print(file)
# str = ''
# for line in file:
#     str = line
# print(str)
#content_str = " ".join(str)
print("  ".join(jieba.analyse.extract_tags(file,topK=10,withWeight=False)))
# -*- coding: UTF-8 -*-
import codecs
import jieba
import jieba.analyse
#添加用户自定义的词典
# jieba.load_userdict("E:/WorkSpace/PycharmProject/Text_cluster/Data/userdict.txt")
#
# file_path = 'E:/WorkSpace/PycharmProject/Text_cluster/Data/Teacher_Data.txt'
# files = codecs.open(file_path,'rb','utf-8').readlines()
# corpus = []
# Lens = []
# # 读取预料 一行预料为一个文档,存于corpus列表中
# for line in files:
#     corpus.append(line.strip())
#     Lens.append(len(line))
# # print(max(Lens))
# # print(min(Lens))
#
# if __name__ == '__main__':
#     for data in corpus:
#         if len(data)>5000:
#             print()