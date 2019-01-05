import gensim
from sklearn.datasets import fetch_20newsgroups
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
import os
from pprint import pprint
news_dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = news_dataset.data
print("In the dataset there are", len(documents), "textual documents")
print ("And this is the first one:\n", documents[0])

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]
print("After the tokenizer, the previous document becomes:\n", tokenize(documents[0]))

# Next step: tokenize all the documents and build a count dictionary, that contains the count of the tokens over the complete text corpus.
processed_docs = [tokenize(doc) for doc in documents]
word_count_dict = Dictionary(processed_docs)
print("In the corpus there are", len(word_count_dict), "unique tokens")

print("\n",word_count_dict,"\n")

word_count_dict.filter_extremes(no_below=20, no_above=0.1)  # word must appear >10 times, and no more than 20% documents
print("After filtering, in the corpus there are only", len(word_count_dict), "unique tokens")

bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]  # bow all document of corpus

model_name = "./model.lda"
if os.path.exists(model_name):
    lda_model = gensim.models.LdaModel.load(model_name)
    print("loaded from old")
else:
    # preprocess()
    lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=100, id2word=word_count_dict, passes=5)#num_topics: the maximum numbers of topic that can provide
    lda_model.save(model_name)
    print("loaded from new")

# 1.
# if you don't assign the target document, then
# every running of lda_model.print_topics(k) gonna get top k topic_keyword from whole the corpora documents in the bag_of_words_corpus from 0-n.
# and if given a single new document, it will only analyse this document, and output top k topic_keyword from this document.

pprint(lda_model.print_topics(30,6))#by default num_topics=10, no more than LdaModel's; by default num_words=10, no limitation
print("\n")
# pprint(lda_model.print_topics(10))

# 2.
# when you assign a particular document for it to assign:
# pprint(lda_model[bag_of_words_corpus[0]].print_topics(10))
for index, score in sorted(lda_model[bag_of_words_corpus[0]], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
print
print(news_dataset.target_names[news_dataset.target[0]])  # bag_of_words_corpus align to news_dataset
print("\n")

# 3.
# process an unseed document
unseen_document = "In my spare time I either play badmington or drive my car"
print("The unseen document is composed by the following text:", unseen_document)
print
bow_vector = word_count_dict.doc2bow(tokenize(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 7)))