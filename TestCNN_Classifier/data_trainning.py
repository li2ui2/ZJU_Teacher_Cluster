import numpy as np
from tensorflow.contrib import learn
from TestCNN_Classifier11 import data_helpers
from TestCNN_Classifier.TextCNNClassifier import NN_config, CALC_config, TextCNNClassifier
# Data Preparation
# ==================================================
positive_data_file = r"E:\WorkSpace\PycharmProject\Text_cluster\Data\CNNdata\rt-polaritydata\rt-polarity.pos"
negative_data_file = r"E:\WorkSpace\PycharmProject\Text_cluster\Data\CNNdata\rt-polaritydata\rt-polarity.neg"
dev_sample_percentage = 0.1
# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

print('vocabulary length is:',len(vocab_processor.vocabulary_))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print('The leangth of X_train is {}'.format(len(x_train)))
print('The length of x_dev is {}'.format(len(x_dev)))


#------------------------------------------------------------------------------
# ----------------  model processing ------------------------------------------
#------------------------------------------------------------------------------
num_seqs = max_document_length
num_classes = 2
num_filters = 128
filter_steps = [5,6,7]
embedding_size = 200
vocab_size = len(vocab_processor.vocabulary_)

learning_rate = 0.001
batch_size    = 128
num_epoches   = 20
l2_ratio      = 0.0

trains = list(zip(x_train, y_train))
devs   = list(zip(x_dev,y_dev))

config_nn = NN_config(num_seqs      = num_seqs,
                      num_classes   = num_classes,
                      num_filters   = num_filters,
                      filter_steps  = filter_steps,
                      embedding_size= embedding_size,
                      vocab_size    = vocab_size)
config_calc = CALC_config(learning_rate = learning_rate,
                          batch_size    = batch_size,
                          num_epoches   = num_epoches,
                          l2_ratio      = l2_ratio)

print('this is checking list:\\\\\n',
        'num_seqs:{}\n'.format(num_seqs),\
        'num_classes:{} \n'.format(num_classes),\
        'embedding_size:{}\n'.format(embedding_size),\
        'num_filters:{}\n'.format(num_filters),\
        'vocab_size:{}\n'.format(vocab_size),\
        'filter_steps:',filter_steps)
print('this is check calc list:\\\\\n',
        'learning_rate :{}\n'.format(learning_rate),\
        'num_epoches: {} \n'.format(num_epoches),\
        'batch_size: {} \n'.format(batch_size),\
        'l2_ratio : {} \n'.format(l2_ratio))


text_model = TextCNNClassifier(config_nn,config_calc)
text_model.fit(trains)
accuracy = text_model.predict_accuracy(devs,test=False)
print('the dev accuracy is :',accuracy)

predictions = text_model.predict(x_dev)
#print(predictions)
