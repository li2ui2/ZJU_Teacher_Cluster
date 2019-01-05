# 停用词表按照行进行存储，每一行只有一个词语
# 将网络上收集到的停用词表去重
def stopwd_reduction(infilepath, outfilepath):
    infile = open(infilepath, 'r', encoding='utf-8')
    outfile = open(outfilepath, 'w')
    stopwordslist = []
    '''
        infile.read().split('\n')：
        read函数读取文本内容为str格式，
        再通过split函数对字符串进行切片，并返回分割后的字符串列表(list)
    '''
    for str in infile.read().split('\n'):
        if str not in stopwordslist:
            stopwordslist.append(str)
            outfile.write(str + '\n')


stopwd_reduction('./test/stopwords.txt', './test/stopword.txt')
