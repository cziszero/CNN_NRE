import tensorflow as tf
import numpy as np
# from tensorflow.contrib.rnn.python.ops import rnn_cell


class Settings(object):

    def __init__(self, MAXMode, ATTMode):

        self.vocab_size = 114044  # 所有的单词数
        self.sen_len = 70  # 句子长度
        self.num_epochs = 5  # 训练epochs次数
        self.num_classes = 53  # 类别数
        self.pos_size = 5  # 位置向量维度
        self.word_size = 50  # 词向量维度
        self.filter_num = 230  # 卷积核个数
        self.filter_height = 3  # 卷积核高度
        self.filter_width = self.word_size + 2 * self.pos_size  # 卷积核宽度
        self.keep_prob = 0.75  # Dropout的保持比
        self.num_layers = 1
        self.pos_num = 123
        self.bag_num = 50  # 每个batch有多少个包 test 262 * 9
        self.ATT_K = 3  # 每个bag中取前K个
        self.MAXMode_normal = 'MAX_Normal'  # 普通的Max-Pooling
        self.MAXMode_TopK = 'MAX_TopK'  # 'TopK' # K-Max-Pooling
        self.MAXMode = MAXMode  # 当前使用什么Max-Pooling模式
        self.ATTMode_one = 'ATT_One'  # 每个bag中只取一个
        self.ATTMode_TopK = 'ATT_opK'  # 每个bag中取前K个
        self.ATTMode_Weight = 'ATT_Weight'  # 每个bag中按权值加和
        self.ATTMode_Avg = 'ATT_Avg'  # 每个bag中句子取平均
        self.ATTMode = ATTMode  # 当前使用什么Max-Pooling模式
        self.model_save_path = 'model/model_%s_%s/' % (
            self.MAXMode, self.ATTMode)
        self.summary_path = 'summary_%s_%s/' % (self.MAXMode, self.ATTMode)
        self.out_path = 'out/out_%s_%s' % (self.MAXMode, self.ATTMode)
        if MAXMode == self.MAXMode_normal:
            self.MaxPool_K = 1  # TopK Max-Pooling 中的K
        elif MAXMode == self.MAXMode_TopK:
            self.MaxPool_K = 3
        else:
            self.MaxPool_K = 300
            print("ERROR MaxPool_k")


class CNN:

    def max_pooling(self, conv):
        if self.setting.MAXMode == self.setting.MAXMode_normal:
            # 简单的Max-Pooling
            fea = tf.nn.max_pool(value=conv, ksize=[
                                 1, self.setting.sen_len - 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            return fea
        elif self.setting.MAXMode == self.setting.MAXMode_TopK:
            # Top K Max-Pooling
            conv = tf.transpose(conv, [0, 2, 3, 1])
            fea, ind = tf.nn.top_k(conv,  self.setting.MaxPool_K)
            fea = tf.reshape(
                fea, [-1,  self.setting.filter_num * self.setting.MaxPool_K])
            return fea
        else:
            print('ERROR!!!!!!!!!! Max-Pooling 模式错误')
            return None

    def getBagFea(self, sen_alpha, sen_repre):
        if self.setting.ATTMode == self.setting.ATTMode_Weight:
            m_tb = tf.matmul(sen_alpha, sen_repre)
        elif self.setting.ATTMode == self.setting.ATTMode_one:
            f, ind = tf.nn.top_k(sen_alpha)
            m_tb = f[0][0] * sen_repre[ind[0][0], :]
        elif self.setting.ATTMode == self.setting.ATTMode_TopK:
            pass
            return None
            # m_att_k_num = tf.minimum(self.setting.att, batch_size)
            # f, ind = tf.nn.top_k(m_ta, m_att_k_num)
            # m_tta, m_ttb = [], []
            # for j in range(m_att_k_num):
            #     m_tta.append(f[:, j])
            #     m_ttb.append(sen_repre[i][ind[0, j], :])
            # m_tta, m_ttb = tf.concat(m_tta, 0), tf.concat(m_ttb, 0)
            # m_tta = tf.reshape(m_tta, (1, -1))
            # m_ttb = tf.reshape(m_ttb, (-1, filter_num * K))
            # m_tb = tf.matmul(m_tta, m_ttb)
        elif self.setting.ATTMode == self.setting.ATTMode_Avg:
            pass
            return None
        return tf.reshape(m_tb, [self.setting.filter_num * self.setting.MaxPool_K, 1])

    def __init__(self, is_training, word_embeddings, settings):

        self.setting = settings

        self.input_word = tf.placeholder(
            dtype=tf.int32, shape=[None, self.setting.sen_len], name='input_word')
        self.input_pos1 = tf.placeholder(
            dtype=tf.int32, shape=[None, self.setting.sen_len], name='input_pos1')
        self.input_pos2 = tf.placeholder(
            dtype=tf.int32, shape=[None, self.setting.sen_len], name='input_pos2')
        self.input_y = tf.placeholder(
            dtype=tf.float32, shape=[None, self.setting.num_classes], name='input_y')
        self.total_shape = tf.placeholder(
            dtype=tf.int32, shape=[self.setting.bag_num + 1], name='total_shape')

        word_embedding = tf.get_variable(
            initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable(
            'pos1_embedding', [self.setting.pos_num, self.setting.pos_size])
        pos2_embedding = tf.get_variable(
            'pos2_embedding', [self.setting.pos_num, self.setting.pos_size])
        B = tf.get_variable(
            'attention_B', [self.setting.sen_len])
        RB1 = tf.get_variable(
            'query_rB1', [self.setting.filter_width, 1])
        # RB2 = tf.get_variable(
        #     'query_rB2', [self.setting.sen_len, 1])
        A = tf.get_variable(
            'attention_A', [self.setting.filter_num * self.setting.MaxPool_K])
        R = tf.get_variable(
            'query_rA', [self.setting.filter_num * self.setting.MaxPool_K, 1])
        relation_embedding = tf.get_variable(
            'relation_embedding', [self.setting.num_classes, self.setting.filter_num * self.setting.MaxPool_K])
        bias = tf.get_variable('bias_d', [self.setting.num_classes])

        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

        # embedding layer
        m_a = tf.nn.embedding_lookup(word_embedding, self.input_word)
        m_b = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        m_c = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)

        m_d = tf.concat(axis=2, values=[m_a, m_b, m_c])  # [None 70 60]

        m_e = tf.multiply(B, tf.transpose(m_d, (0, 2, 1)))  # BX
        m_e = tf.transpose(m_e, (0, 2, 1))
        m_f = tf.reshape(tf.matmul(tf.reshape(
            m_e, (-1, self.setting.filter_width)), RB1), (-1, self.setting.sen_len))
        m_g = tf.reshape(m_f, (-1, self.setting.sen_len))
        m_h = tf.nn.softmax(m_g)
        m_i = tf.matrix_diag(m_h)
        input_x = tf.matmul(m_i, m_d)
        print("m_e", m_e)
        print("m_f", m_f)
        print("m_g", m_g)
        print("m_h", m_h)
        print("m_i", m_i)
        print("input_x", input_x)

        filters = tf.get_variable('filters', [
                                  self.setting.filter_height, self.setting.filter_width, self.setting.filter_num])
        conv = tf.nn.conv1d(value=input_x, filters=filters,
                            stride=1, padding='VALID')
        conv = tf.reshape(
            conv, [-1, self.setting.sen_len - 2, 1,  self.setting.filter_num])  # 'NHWC'

        fea = self.max_pooling(conv)

        # sentence-level attention layer
        for i in range(self.setting.bag_num):

            # sen_repre 表示这一bag中句子的特征向量矩阵
            sen_repre = tf.tanh(
                fea[self.total_shape[i]:self.total_shape[i + 1]])
            # 这个bag中有多少个句子
            batch_size = self.total_shape[i + 1] - self.total_shape[i]
            # 计算权值
            sen_repre = tf.reshape(
                sen_repre, (-1, self.setting.filter_num * self.setting.MaxPool_K))
            m_te = tf.multiply(sen_repre, A)  # xA

            # print("***sen_repre***", sen_repre)
            # print("***A***", A)
            # return(0)
            m_td = tf.matmul(m_te, R)  # xAR
            m_tc = tf.reshape(m_td, [batch_size])
            m_tb = tf.nn.softmax(m_tc)  # softmax(xAR)
            # sen_alpha表达第这个袋中的权值
            sen_alpha = tf.reshape(m_tb, [1, batch_size])

            # sen_s就是最后第i个袋的特征
            sen_s = self.getBagFea(sen_alpha=sen_alpha, sen_repre=sen_repre)

            if is_training and self.setting.keep_prob < 1:
                sen_s = tf.nn.dropout(sen_s, self.setting.keep_prob)

            # sen_out即第i个bag对类别的打分
            m_ta = tf.matmul(relation_embedding, sen_s)
            m_tb = tf.reshape(m_ta, [self.setting.num_classes])
            sen_out = tf.add(m_tb, bias)

            # prob[i]即第i个bag属于第i类的概率
            self.prob.append(tf.nn.softmax(sen_out))

            # 预测值
            with tf.name_scope("output"):
                self.predictions.append(
                    tf.argmax(self.prob[i], 0, name="predictions"))

            # 计算交叉熵
            with tf.name_scope("loss"):
                m_ta = tf.nn.softmax_cross_entropy_with_logits(
                    logits=sen_out, labels=self.input_y[i])
                self.loss.append(tf.reduce_mean(m_ta))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            with tf.name_scope("accuracy"):
                m_tb = tf.argmax(self.input_y[i], 0)
                m_ta = tf.equal(self.predictions[i], m_tb)
                m_tc = tf.cast(m_ta, "float")
                self.accuracy.append(tf.reduce_mean(m_tc, name="accuracy"))

        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(0.0001), weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)
