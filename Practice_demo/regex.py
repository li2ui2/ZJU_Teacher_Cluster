# encoding: UTF-8
import re
def main(str):
    # 将正则表达式编译成Pattern对象
    pattern = re.compile(r'hello.*\!')
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    match = pattern.match(str)
    if match:
        #使用Match获得分组信息
        print(match.group())
    # content = 'hello, liwei! how are you.'
    # pattern = re.compile('hello.*\!')
    # match = re.match(pattern,content)
    # print(match.group())
def fenge():
    p = re.compile('\d+')
    print(p.split('one1two2three3four4'))
def sousuo():
    p = re.compile('\d+')
    #搜索string，以列表形式返回全部能匹配的字串
    print(p.findall('one1two2three3four4'))
def tihuan():
    p = re.compile(r'(\w+) (\w+)')
    s = 'i say, hello liwei'
    print(p.sub(r'\2 \1', s)) #将分组1替换为分组2

if __name__ == "__main__":
    content = 'hello, liwei! How are you'
    main(content)

    # 分割字符串
    fenge()

    # 搜索string，以列表形式返回全部能匹配的字串
    sousuo()
    tihuan()

