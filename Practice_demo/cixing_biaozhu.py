# -*- coding: utf-8 -*-

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# 词性标注器
text = word_tokenize("And now for something completely different")
print(pos_tag(text))

text = word_tokenize("They refuse to permit us to obtain the refuse permit")
pos_tag(text)

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')   #查词性相同的词语
text.similar('bought')
text.similar('over')
text.similar('the')

# 标注语料库
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
print(tagged_token[0])
print(tagged_token[1])

sent = '''
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PP
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/R
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.
 '''
[nltk.tag.str2tuple(t) for t in sent.split()]

nltk.corpus.brown.tagged_words()

print(nltk.corpus.nps_chat.tagged_words())
print(nltk.corpus.conll2000.tagged_words())
print(nltk.corpus.treebank.tagged_words())

#统一标注方式，方便联合使用
print(nltk.corpus.brown.tagged_words(tagset='universal'))
print(nltk.corpus.nps_chat.tagged_words(tagset='universal'))
print(nltk.corpus.conll2000.tagged_words(tagset='universal'))

print(nltk.corpus.treebank.tagged_words(tagset='universal'))

print(nltk.corpus.sinica_treebank.tagged_words())
print(nltk.corpus.indian.tagged_words())
print(nltk.corpus.mac_morpho.tagged_words())
print(nltk.corpus.conll2002.tagged_words())
print(nltk.corpus.cess_cat.tagged_words())

from nltk.corpus import brown

brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()  #显示最常出现的词性
tag_fd.plot()



#常见词性的句法结构
# 名词
word_tag_pairs = list(nltk.bigrams(brown_news_tagged))
nltk.FreqDist(a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN').most_common()

# 动词
wsj = list(nltk.corpus.treebank.tagged_words(tagset='universal'))
word_tag_fd = nltk.FreqDist(wsj).most_common()
[wordtag[0] + "/" + wordtag[1] for (wordtag, fred) in word_tag_fd if wordtag[1].startswith('VERB')][:50]

cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].keys() #查找单词词性
cfd1['cut'].keys()

cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
cfd2['PRT'].most_common(10)


# 未简化标记
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())


tagdict = findtags('NN', brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
    print(tag, tagdict[tag])

# 搜索已标注的语料库
brown_learned_text = nltk.corpus.brown.tagged_words(categories='learned')
sorted(set(b[0] for (a, b) in list(nltk.bigrams(brown_learned_text)) if a[0] == 'often'))

#查找最常见的词性
brown_lrnd_tagged = nltk.corpus.brown.tagged_words(tagset='universal')
tags = [b[1] for (a, b) in list(nltk.bigrams(brown_lrnd_tagged)) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()



from nltk.corpus import brown

#查找'V' To 'V'结构的三个词
def process(sentence):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)
for tagged_sent in brown.tagged_sents():
    process(tagged_sent)



brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag)
                                for (word, tag) in brown_news_tagged)
for word in data.conditions():
    if len(data[word]) > 3:
        tags = data[word].keys()
        print(word, ' '.join(tags))

####自动标记####
# 默认标注器
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()

raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')  #给每个词都标注为NN
default_tagger.tag(tokens)

default_tagger.evaluate(brown.tagged_sents(categories='news'))

# 正则表达式标注器
patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown.sents()[3])

regexp_tagger.evaluate(brown.tagged_sents(categories='news'))

# 查询标注器
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common()[:100]
likely_tags = dict((word, cfd[word].max()) for (word, freq) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent)

baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))


def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))


def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()


display()

####N-gram标注####
# 一元模型
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
print(unigram_tagger.evaluate(brown_tagged_sents))

# 分离训练与测试数据
size = int(len(brown_tagged_sents) * 0.9)
print(size)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

# 一般N-gram标注
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])

unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)

bigram_tagger.evaluate(test_sents)

# 组合标注器
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)

# 储存标注器
from pickle import dump

output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

from pickle import load

input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

text = """The board's action shows what free enterprise
    is up against in our complex maze of regulatory laws ."""
tokens = text.split()
tagger.tag(tokens)

# 性能限制
cfd = nltk.ConditionalFreqDist(
    ((x[1], y[1], z[0]), z[1])
    for sent in brown_tagged_sents
    for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
sum(cfd[c].N() for c in ambiguous_contexts) / float(cfd.N())

test_tags = [tag for sent in brown.sents(categories='editorial')
             for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print(nltk.ConfusionMatrix(gold_tags, test_tags))

# 跨句子边界标注
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t2.evaluate(test_sents)

