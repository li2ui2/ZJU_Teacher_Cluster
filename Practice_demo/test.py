# -*-coding: utf-8 -*-
# import re
# #pattern = re.compile('[^\u4e00-\u9fa5]')
# pattern = re.compile('[^\b\d+(?:\.\d+)\s+]')
# name = '2018中111d国人1厉害qwertyuioplkjh5464gfdsazxcvbnm1231564648'
# b = pattern.sub('',name)
# print(b)
# #去掉文本行里面的空格、\t、数字（其他有要去除的也可以放到' \t1234567890'里面）
# print(list(filter(lambda x:x not in '0123456789',name)))
import jieba
import jieba.analyse
import os
import re
import codecs


def segment_lines(source_in_dir, segment_out_dir, stopwords=[]):
    file_list = os.listdir(source_in_dir)
    for file in file_list:
        source =os.path.join(source_in_dir,file)
        with open(source, 'rb') as f:
            document = f.read()
            document = filter(lambda ch: ch not in '0123456789', str(document))
            document = ''.join(list(document))
            with codecs.open(segment_out_dir, 'wb',encoding='utf-8') as f2:
                f2.write(document)
source_in_dir= 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\source'
segment_out_dir = 'E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\Teacher_Data_cnum.txt'
segment_lines(source_in_dir, segment_out_dir, stopwords=[])