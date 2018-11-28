# coding=utf-8

from DatasetLoader import Dataset
from tfConf import getSess

import tensorflow as tf


class Param():
    MAX_SEN_LEN = 86
    BATCH_SIZE = 40
    WF_SIZE = 50
    PF_SIZE = 5
    CONV_FILTERS = 230
    CONV_HIGHT = 3
    REL_NUM = 19
    LR = 0.001


class Model():
    def __init__(self, sess, encoder, drop_prob=0.5):
        self.encoder = encoder
        self.sess = sess

        self.inputs = tf.placeholder(shape=(Param.BATCH_SIZE, Param.MAX_SEN_LEN, Param.WF_SIZE + 2 * Param.PF_SIZE),
                                     dtype=tf.float32, name='input')
        self.training = tf.placeholder(dtype=tf.bool, name='training')
        self.label = tf.placeholder(shape=(Param.BATCH_SIZE), dtype=tf.int32, name='label')

        conv = tf.layers.conv1d(self.inputs, filters=Param.CONV_FILTERS, kernel_size=Param.CONV_HIGHT, padding='same')
        if encoder == 'CNN':
            _f = self._cnn_encoder(conv)
        elif encoder == 'CNN-K':
            _f = self._cnn_k_encoder(conv)
        elif encoder == 'PCNN':
            _f = self._pcnn_encoder(conv)
        elif encoder == 'PCNN-K':
            _f = self._pcnn_k_encoder(conv)

        act = tf.nn.relu(_f)
        fea = tf.layers.dropout(act, rate=drop_prob, training=self.training)

        logit = tf.layers.dense(fea, Param.REL_NUM, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.pred = tf.argmax(logit, axis=1)

        label_onehot = tf.one_hot(indices=self.label, depth=Param.REL_NUM, dtype=tf.int32)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=logit)
        self.train_op = tf.train.AdamOptimizer(Param.LR).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)

        self.sum_ops = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

    def _cnn_encoder(self, conv):
        gmax = tf.reduce_max(conv, axis=1)
        return gmax

    def _cnn_k_encoder(self, conv):
        k = 2
        p, _ = tf.nn.top_k(tf.transpose(conv, [0, 2, 1]), k)
        return tf.reshape(p, (-1, conv.get_shape().as_list()[-1] * k))

    def _pcnn_encoder(self, conv):
        self.pos = tf.placeholder(shape=(Param.BATCH_SIZE, 2), dtype=tf.int32, name='pos')
        pmax = []
        for i in range(Param.BATCH_SIZE):
            e1, e2 = self.pos[i][0], self.pos[i][1]
            p1 = tf.reduce_max(conv[i, 0:e1, :], axis=0)
            p2 = tf.reduce_max(conv[i, e1:e2, :], axis=0)
            p3 = tf.reduce_max(conv[i, e2:Param.MAX_SEN_LEN, :], axis=0)
            p = tf.concat([p1, p2, p3], axis=0)
            pmax.append(p)
        return tf.stack(pmax)

    def _pcnn_k_encoder(self, conv):
        self.pos = tf.placeholder(shape=(Param.BATCH_SIZE, 2), dtype=tf.int32, name='pos')
        padding = tf.zeros(dtype=tf.float32, shape=(1, conv.get_shape().as_list()[-1]))
        pmax = []
        for i in range(Param.BATCH_SIZE):
            e1, e2 = self.pos[i][0], self.pos[i][1]

            _p = tf.concat([conv[i, 0:e1, :], padding], axis=0)
            p1, k = tf.nn.top_k(tf.transpose(_p), k=2)

            _p = tf.concat([conv[i, e1:e2, :], padding], axis=0)
            p2, k = tf.nn.top_k(tf.transpose(_p), k=2)

            _p = tf.concat([conv[i, e2:Param.MAX_SEN_LEN, :], padding], axis=0)
            p3, k = tf.nn.top_k(tf.transpose(_p), k=2)

            p = tf.reshape(tf.concat([p1, p2, p3], axis=0), [-1])
            pmax.append(p)
        return tf.stack(pmax)

    def train(self, x_bat, y_bat, pos_bat):
        feed = {self.inputs: x_bat, self.label: y_bat, self.training: True}
        if 'PCNN' in self.encoder:
            feed[self.pos] = pos_bat
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed)
        return loss

    def test(self, x_bat, y_bat, pos_bat):
        feed = {self.inputs: x_bat, self.training: False}
        if 'PCNN' in self.encoder:
            feed[self.pos] = pos_bat
        pred = self.sess.run(self.pred, feed_dict=feed)
        return sum(pred == y_bat)


def train_test(train_data, test_data, encoder):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    sess = getSess()

    model = Model(sess, encoder=encoder)

    # summary_writer = tf.summary.FileWriter('summary', sess.graph)
    loss, cor = 0.0, 0
    for x, y, pos in test_data:
        cor += model.test(x, y, pos)
    print('random initialization, accuracy rate %f' % (cor / test_data.size))

    for epoch in range(1, 61):
        loss, cor = 0.0, 0
        for x, y, pos in train_data:
            loss += model.train(x, y, pos)
            # summary_writer.add_summary(sess.run(model.sum_ops))
        print('epoch %d, loss %f' % (epoch, loss / (train_data.size / Param.BATCH_SIZE)))
        for x, y, pos in test_data:
            cor += model.test(x, y, pos)
        print('epoch %d, accuracy rate %f' % (epoch, cor / test_data.size))

    del train_data, test_data
    sess.close()


if __name__ == '__main__':
    train_data = Dataset(data_path='data/train_file.txt', isRandom=False)
    test_data = Dataset(data_path='data/test_file.txt', isRandom=False)

    # 使用Word2Vec初始化 PCNN结构
    st = '使用Word2Vec初始化 PCNN结构'
    print('=' * 10, st, '=' * 10)
    train_test(train_data, test_data, encoder='PCNN')

    # st = '使用Word2Vec初始化 PCNN-K结构'
    st = '使用Word2Vec初始化 PCNN-K结构'
    print('=' * 10, st, '=' * 10)
    train_test(train_data, test_data, encoder='PCNN-K')

    st = '使用Word2Vec初始化 CNN-K结构'
    print('=' * 10, st, '=' * 10)
    train_test(train_data, test_data, encoder='CNN-K')

    # 使用Word2Vec初始化 CNN结构
    st = '使用Word2Vec初始化 CNN结构'
    print('=' * 10, st, '=' * 10)
    train_test(train_data, test_data, encoder='CNN')

    # train_data = Dataset(data_path='data/train_file.txt', isRandom=True)
    # test_data = Dataset(data_path='data/test_file.txt', isRandom=True)

    # 使用Random初始化 PCNN结构
    # st = '使用Random初始化 PCNN结构'
    # print('=' * 10, st, '=' * 10)
    # train_test(train_data, test_data, encoder='PCNN')

    # 使用Random初始化 CNN结构
    # st = '使用Random初始化 CNN结构'
    # print('=' * 10, st, '=' * 10)
    # train_test(train_data, test_data, encoder='CNN')
