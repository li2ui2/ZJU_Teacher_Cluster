# coding: utf-8
import tensorflow as tf
import numpy as np
import os

class NN_config(object):
    def __init__(self, vocab_size, num_filters,filter_steps,num_seqs=1000,num_classes=2, embedding_size=200):
        self.vocab_size     = vocab_size
        self.num_filters    = num_filters
        self.filter_steps   = filter_steps
        self.num_seqs       = num_seqs
        self.num_classes    = num_classes
        self.embedding_size = embedding_size

class CALC_config(object):
    def __init__(self, learning_rate=0.0075, batch_size=64, num_epoches=20, l2_ratio=0.0):
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.num_epoches   = num_epoches
        self.l2_ratio      = l2_ratio

class TextCNNClassifier(object):
    '''
    A class used to define text classifier use convolution network
    the form of class like keras or scikit-learn
    '''
    def __init__(self,config_nn, config_calc):

        self.num_seqs = config_nn.num_seqs
        self.num_classes = config_nn.num_classes
        self.embedding_size = config_nn.embedding_size
        self.vocab_size  = config_nn.vocab_size
        self.num_filters = config_nn.num_filters
        self.filter_steps = config_nn.filter_steps

        self.learning_rate = config_calc.learning_rate
        self.batch_size    = config_calc.batch_size
        self.num_epoches   = config_calc.num_epoches
        self.l2_ratio      = config_calc.l2_ratio

        #tf.reset_default_graph()
        self.build_placeholder()
        self.build_embedding_layer()
        self.build_nn()
        self.build_cost()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_placeholder(self):
        with tf.name_scope('inputs_to_data'):
            self.inputs       = tf.placeholder( tf.int32,shape=[None, self.num_seqs],name='inputs')
            self.targets      = tf.placeholder(tf.float32,shape=[None, self.num_classes], name='targets')
            self.keep_prob    = tf.placeholder(tf.float32, name='nn_keep_prob')
            print('self.inputs.shape:',self.inputs.shape)

    def build_embedding_layer(self):
        with tf.device('/cpu:0'),tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_size],stddev=0.1),\
                        name = 'embeddings')
            x = tf.nn.embedding_lookup(embeddings, self.inputs)
            x = tf.expand_dims(x, axis=-1)
            self.x = tf.cast(x, tf.float32 )
            print('x shape is:',self.x.get_shape())

    def build_nn(self):
        conv_out = []
        for i , filter_step in enumerate(self.filter_steps):
            with tf.name_scope("conv-network-%s"%filter_step):
                filter_shape = [filter_step,self.embedding_size, 1,self.num_filters]
                filters = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1), \
                          name='filters')
                bias    = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name='bias')
                # h_conv : shape =batch_szie * (num_seqs-filter_step+1) * 1 * num_filters
                h_conv  = tf.nn.conv2d(self.x,
                                    filter=filters,
                                    strides = [1,1,1,1],
                                    padding='VALID',
                                    name='hidden_conv')
                h_relu  = tf.nn.relu(tf.nn.bias_add(h_conv,bias),name='relu')
                ksize = [1,self.num_seqs-filter_step+1,1,1]
                #h_pooling: shape = batch_size * 1 * 1 * num_filters
                h_pooling = tf.nn.max_pool(h_relu,
                                           ksize=ksize,
                                           strides=[1,1,1,1],
                                           padding='VALID',
                                           name='pooling')
                conv_out.append(h_pooling)

        self.tot_filters_units = self.num_filters * len(self.filter_steps)
        self.h_pool = tf.concat(conv_out,axis=3)
        self.h_pool_flattern =tf.reshape(self.h_pool, shape=[-1, self.tot_filters_units])

        with tf.name_scope('dropout'):
            self.h_pool_drop = tf.nn.dropout(self.h_pool_flattern, self.keep_prob)

    def build_cost(self):
        with tf.name_scope('cost'):
            W = tf.get_variable(shape=[self.tot_filters_units, self.num_classes],name='W',\
                    initializer = tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes],name='bias'))
            self.scores =  tf.nn.xw_plus_b(self.h_pool_drop, W, bias, name='scores')
            self.predictions = tf.argmax(self.scores,axis=1,name='predictions')
            l2_loss = tf.constant(0.0,name='l2_loss')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(bias)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.targets)
            self.loss = tf.reduce_mean(losses) + self.l2_ratio*l2_loss

        with tf.name_scope('accuracy'):
            pred = tf.equal(self.predictions, tf.argmax(self.targets, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))

    def build_optimizer(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grad_and_vars)

    def random_batches(self,data,shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((data_size-1)/self.batch_size) + 1
        if shuffle :
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_index]
        else:
            shuffled_data = data
        #del data
        for epoch in range(self.num_epoches):
            for batch_num in range(num_batches_per_epoch):
                start = batch_num * self.batch_size
                end   = min(start + self.batch_size,data_size)
                yield shuffled_data[start:end]

    def fit(self,data):
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        self.session = tf.Session()
        with self.session as sess:
            #self.saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())
            batches = self.random_batches(list(data))
            accuracy_list = []
            loss_list = []
            #prediction_list = []
            iterations = 0
            # model saving
            save_path = os.path.abspath(os.path.join(os.path.curdir, 'models'))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for batch in batches:
                iterations  += 1
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                feed = { self.inputs: x_batch,
                         self.targets: y_batch,
                         self.keep_prob: 0.5}
                batch_pred, batch_accuracy, batch_cost, _ = sess.run([self.predictions, self.accuracy,\
                                                        self.loss, self.train_op], feed_dict=feed)
                accuracy_list.append(batch_accuracy)
                loss_list.append(batch_cost)

                if iterations % 10 == 0:
                    print('The trainning step is {0}'.format(iterations),\
                         'trainning_loss: {:.3f}'.format(loss_list[-1]), \
                         'trainning_accuracy: {:.3f}'.format(accuracy_list[-1]))
                if iterations % 100 == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step = iterations)

            self.saver.save(sess, os.path.join(save_path, 'model'), global_step = iterations)

    def load_model(self,start_path=None):
        if start_path == None:
            start_path = os.path.abspath(os.path.join(os.path.curdir,'models'))
            print('default start_path is',start_path)
            #star = start_path
            ckpt = tf.train.get_checkpoint_state('./models')
            print('This is out checking of ckpt:',ckpt.model_checkpoint_path)
            #self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            self.session = tf.Session()
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print('Restored from {} completed'.format(ckpt.model_checkpoint_path))
        else:
            self.session = tf.Session()
            self.saver.restore(self.session,start_path)
            print('Restored from {} completed'.format(start_path))

    def predict_accuracy(self,data,test=True):
        # loading_model
        self.load_model()
        sess = self.session
        iterations = 0
        accuracy_list = []
        predictions = []
        self.num_epoches = 1
        batches = self.random_batches(data,shuffle=False)
        for batch in batches:
            iterations += 1
            x_inputs, y_inputs = zip(*batch)
            x_inputs = np.array(x_inputs)
            y_inputs = np.array(y_inputs)
            feed = {self.inputs: x_inputs,
                    self.targets: y_inputs,
                    self.keep_prob: 1.0
                }
            batch_pred, batch_accuracy, batch_loss = sess.run([self.predictions,\
            self.accuracy, self.loss], feed_dict=feed)
            accuracy_list.append(batch_accuracy)
            predictions.append(batch_pred)
            print('The trainning step is {0}'.format(iterations),\
                 'trainning_accuracy: {:.3f}'.format(accuracy_list[-1]))

        accuracy = np.mean(accuracy_list)
        predictions = [list(pred) for pred in predictions]
        predictions = [p for pred in predictions for p in pred]
        predictions = np.array(predictions)
        if test :
            return predictions, accuracy
        else:
            return accuracy

    def predict(self, data):
        # load_model
        self.load_model()
        sess = self.session
        iterations = 0
        predictions_list = []
        self.num_epoches = 1
        batches = self.random_batches(data)
        for batch in batches:
            x_inputs = batch
            feed = {self.inputs : x_inputs,
                    self.keep_prob:1.0}
            batch_pred = sess.run([self.predictions],feed_dict=feed)
            predictions_list.append(batch_pred)
        predictions = [list(pred) for pred in predictions_list]
        predictions = [p for pred in predictions for p in pred]
        predictions = np.array(predictions).reshape(-1,1)
        print(predictions)
        return predictions
