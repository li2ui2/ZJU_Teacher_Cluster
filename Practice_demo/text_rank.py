import numpy as np
import jieba
import jieba.posseg as pseg


class TextRank(object):

    def __init__(self, sentence, window, alpha, iternum):
        self.sentence = sentence
        self.window = window
        self.alpha = alpha
        self.edge_dict = {}  # 记录节点的边连接字典
        self.iternum = iternum  # 迭代次数

    # 对句子进行分词
    def cutSentence(self):
        jieba.load_userdict('user_dict.txt')
        tag_filter = ['a', 'd', 'n', 'v']
        seg_result = pseg.cut(self.sentence)
        self.word_list = [s.word for s in seg_result if s.flag in tag_filter]
        print(self.word_list)

    # 根据窗口，构建每个节点的相邻节点,返回边的集合
    def createNodes(self):
        tmp_list = []
        word_list_len = len(self.word_list)
        for index, word in enumerate(self.word_list):
            if word not in self.edge_dict.keys():
                tmp_list.append(word)
                tmp_set = set()
                left = index - self.window + 1  # 窗口左边界
                right = index + self.window  # 窗口右边界
                if left < 0: left = 0
                if right >= word_list_len: right = word_list_len
                for i in range(left, right):
                    if i == index:
                        continue
                    tmp_set.add(self.word_list[i])
                self.edge_dict[word] = tmp_set

    # 根据边的相连关系，构建矩阵
    def createMatrix(self):
        self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
        self.word_index = {}  # 记录词的index
        self.index_dict = {}  # 记录节点index对应的词

        for i, v in enumerate(set(self.word_list)):
            self.word_index[v] = i
            self.index_dict[i] = v
        for key in self.edge_dict.keys():
            for w in self.edge_dict[key]:
                self.matrix[self.word_index[key]][self.word_index[w]] = 1
                self.matrix[self.word_index[w]][self.word_index[key]] = 1
        # 归一化
        for j in range(self.matrix.shape[1]):
            sum = 0
            for i in range(self.matrix.shape[0]):
                sum += self.matrix[i][j]
            for i in range(self.matrix.shape[0]):
                self.matrix[i][j] /= sum

    # 根据textrank公式计算权重
    def calPR(self):
        self.PR = np.ones([len(set(self.word_list)), 1])
        for i in range(self.iternum):
            self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)

    # 输出词和相应的权重
    def printResult(self):
        word_pr = {}
        for i in range(len(self.PR)):
            word_pr[self.index_dict[i]] = self.PR[i][0]
        res = sorted(word_pr.items(), key=lambda x: x[1], reverse=True)
        print(res)


if __name__ == '__main__':
    s = '程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
    tr = TextRank(s, 3, 0.85, 700)
    tr.cutSentence()
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    tr.printResult()
