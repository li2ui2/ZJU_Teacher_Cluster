import jieba
import jieba.analyse
import os
import re
import codecs
from string import digits
'''
read() 每次读取整个文件，它通常将读取到底文件内容放到一个字符串变量中，也就是说 .read() 生成文件内容是一个字符串类型。
readline()每次只读取文件的一行，通常也是读取到的一行内容放到一个字符串变量中，返回str类型。
readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。
'''
def getStopwords(path):
    """
    加载停用词
    :param path: txt形式的文件
    :return stopwords: 停用词列表
    """
    print("正在加载停用词...")
    stopwords = []
    with open(path, "r", encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

def clearTXTNum(source_in_path, source_out_path):
    """
    去除文本中的数字
    :param source_in_path:
    :param source_out_path:
    """
    print("正在去除文本中的数字...")
    infile = open(source_in_path, 'r',encoding='utf-8')
    outfile = open(source_out_path, 'w',encoding='utf-8')
    for eachline in infile.readlines():
        remove_digits = str.maketrans('', '', digits)
        lines = eachline.translate(remove_digits)
        #lines.encode('utf-8')
        outfile.write(lines)

def segment_line(file_list,segment_out_dir,stopwords=[]):
    '''
    字词分割，对每行进行字词分割
    :param file_list:
    :param segment_out_dir:
    :param stopwords:
    :return:
    '''
    print("正在进行分词...")
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        segment_file = open(segment_out_name, 'a', encoding='utf8')
        with open(file, encoding='utf8') as f:
            text = f.readlines()
            for sentence in text:
                # jieba.cut():参数sentence必须是str(unicode)类型
                sentence = list(jieba.cut(sentence))
                sentence = re.sub()
                sentence_segment = []
                for word in sentence:
                    if word not in stopwords:
                        sentence_segment.append(word)
                segment_file.write(" ".join(sentence_segment))
            del text
            f.close()
        segment_file.close()

def segment_lines(source_in_dir, segment_out_dir, stopwords=[]):
    '''
    字词分割，对整个文件内容进行字词分割
    :param file_list:
    :param segment_out_dir:
    :param stopwords:
    :return:
    '''
    print("正在进分词...")
    file_list = os.listdir(source_in_dir)
    for file in file_list:
        source =os.path.join(source_in_dir,file)
        with open(source, 'rb') as f:
            document = f.read()
            # document_decode = document.decode('GBK')
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_dir, 'wb') as f2:
                f2.write(result)

def get_keyword(path, keyword_path):
    """
    提取文档中每一行的关键词
    :param path:
    :param keyword_path:
    """
    print("正在提取文档中每一行的关键词...")
    # 引入提取关键词算法的抽取接口（TF-IDF、textrank）
    tfidf = jieba.analyse.extract_tags
    #textrank = jieba.analyse.textrank

    #基于TF-IDF算法进行关键词抽取
    contents = codecs.open(path, 'rb', encoding='utf-8').readlines()
    count = 1
    with codecs.open(keyword_path, 'wb', encoding='utf-8') as f:
        for content in contents:
            #tfidf的输入为str，得到的结果为一个list
            keyword = " ".join(tfidf(content,topK=150,withWeight=False))
            if count == 1:
                f.write((keyword))
            else:
                f.write(('\n' + keyword))
            count += 1
        f.close()

if __name__ == '__main__':
    #文本清洗：去除数字
    source_in_path = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\other data\Teacher_Data.txt'
    source_out_path = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\source\Teachr_Data_delNum.txt'
    clearTXTNum(source_in_path, source_out_path)

    # 多线程分词，windows下暂时不支持
    #jieba.enable_parallel()
    #加载自定义词典
    print("正在加载自定义词典...")
    userdict_path = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\other data\userdict.txt'
    jieba.load_userdict(userdict_path)

    #加载停用词
    stopwords_path = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\other data\stopwords.txt'
    stopwords = getStopwords(stopwords_path)

    #实现分词
    source_in_dir = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\source'
    segment_out_dir = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\segmen_result\seg_result_new.txt'
    segment_lines(source_in_dir, segment_out_dir, stopwords)

    #提取分词结果中的关键词，使用TF-IDF
    keyword_path = r'E:\WorkSpace\PycharmProject\Text_cluster\Data\keword_result\keyword_result_new.txt'
    get_keyword(segment_out_dir, keyword_path)
