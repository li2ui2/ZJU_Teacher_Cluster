import pymongo
import csv
import sys
import os

client = pymongo.MongoClient('localhost',27017)
ZJU_Teacher_Infomation= client['ZJU_Teacher_Infomation']
Teacher_detail_Info = ZJU_Teacher_Infomation['Teacher_detail_Info']

def solve_largeCSVfile():
    """
    以下代码为解决csv文件中数据过大导致无法读取的问题
    """
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True
def text_save(filename, data):
    """
     该函数主要实现list写到txt文件的操作
    :param fileneme: 文件路径/文件名
    :param data: list
    """
    file = open(filename, 'w', encoding = 'utf-8')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')   #列表中特殊字符的处理
        s = s.replace("'", '').replace(',', '').replace("\n",'').replace("null",'')
        file.write(s)
    file.close()
def Text_Union():
    # 获取目标文件夹的路径
    filedir = "E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\Teacher_Data2\\"
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(filedir)
    # 打开当前目录下的result.txt文件，如果没有则创建
    f = open('E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\Teacher_Data.txt', 'w', encoding='utf-8')
    # 先遍历文件名
    for filename in filenames:
        filepath = filedir + '/' + filename
        # 遍历单个文件，读取行数
        for line in open(filepath, 'r', encoding='utf-8'):
            f.writelines(line)
            f.write('\n')
    # 关闭文件
    f.close()
with open("E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\Teacher_detail_Iofo.csv","r",encoding="utf-8") as  csvfile:
    solve_largeCSVfile()
    reader = csv.reader(csvfile)
    sum = 0
    Text_Name = []
    for line in reader:
        #print(f"# line = {line}, typeOfLine = {type(line)}, lenOfLine = {len(line)}")
        del line[0]
        file_name = "E:\\WorkSpace\\PycharmProject\\Text_cluster\\Data\\Teacher_Data2\\" + str(line[0]) + ".txt"
        Text_Name.append(line[0])
        text_save( file_name, line)
        sum += 1
    print("成功存储"+ str(sum) + "个TXT文件")
    print(Text_Name)
    print(len(Text_Name))
Text_Union()


