# coding: utf-8
from constant import *
from m_utils import *

import numpy as np
import gensim
import pickle as pk
import os

# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()


def getWF(word, WF_size, g_w2v, m_w2v, lemmatizer=None):
    # 测试随机生成词向量与word2vec训练向量的差异
    # sen_vec.append(np.random.normal(size=WF_size))
    try:
        return g_w2v[word][0:WF_size]
    except KeyError:
        # try:
        #     a = lemmatizer.lemmatize(word)
        #     return g_w2v[word][0:WF_size]
        # except:
        try:
            # print(word)
            with open('not_in_google.txt', 'a') as f:
                f.write(word + '\n')
            return m_w2v[word][0:WF_size]
        except KeyError:
            print('*' * 10 + word)
            return None


def loadW2vModel():
    p = './data/GoogleNews-vectors-negative300.bin'
    g_w2v = gensim.models.KeyedVectors.load_word2vec_format(p, binary=True)
    hps = {'hs': 0, 'm_iter': 500, 'sg': 0,
           'size': 300, 'window': 3}  # 直接使用词向量分类时的最好词典
    m_w2v = gensim.models.Word2Vec.load('data/' + getOutfilename(hps))
    return g_w2v, m_w2v


def mk_sens_vec(israndom, sens, fn_sens_vec, fn_words_vec):
    if not os.path.exists(fn_words_vec):
        words_vec = {}
    else:
        words_vec = pk.load(open(fn_words_vec, 'rb'))
    # lemmatizer = WordNetLemmatizer()
    sens_vec = []
    for sen in sens:
        sen_vec = []
        sen_vec.append(F_START)
        for word in sen:
            if words_vec.get(word) != None:  # 首先查表
                sen_vec.append(words_vec[word][0:WF_size])
            elif not israndom:  # 使用W2V初始化
                if 'g_w2v' not in dir():
                    g_w2v, m_w2v = loadW2vModel()
                sen_vec.append(getWF(word, WF_size, g_w2v, m_w2v))
                words_vec[word] = sen_vec[len(sen_vec) - 1]
            else:  # 使用随机初始化
                sen_vec.append(np.random.normal(size=(WF_size,)))
                words_vec[word] = sen_vec[len(sen_vec) - 1]
        sen_vec.append(F_END)
        while len(sen_vec) < sen_len:  # 填充至最大句子长度
            sen_vec.append(F_PAD)
        sens_vec.append(sen_vec)

    if 'g_w2v' in dir():
        del g_w2v, m_w2v

    pk.dump(words_vec, open(fn_words_vec, 'wb'))
    pk.dump(sens_vec, open(fn_sens_vec, 'wb'))
    return sens_vec


def get_sens_vec(isPCNN, isRandom, sens, pos,  fn_words_vec, fn_sens_vec):
    """
    return np_train_set
    """
    if os.path.exists(fn_sens_vec):
        sens_vec = pk.load(open(fn_sens_vec, 'rb'))
    else:
        sens_vec = mk_sens_vec(isRandom, sens, fn_sens_vec, fn_words_vec)
    SF = []  # 句子级别特征的向量矩阵
    WLF = []  # 词级别特征
    t_PAD = np.zeros(shape=WF_size * window_size + PF_size * 2, dtype=float)
    for k in range(len(pos)):
        e, sen = pos[k], sens_vec[k]
        SF_one = []
        if isPCNN:
            sen.insert(e[0] + 1, F_E1)
            sen.insert(e[1] + 2, F_E2)
        for i in range(1, len(sen) - 1):
            _ = np.concatenate((sen[i - 1], sen[i], sen[i + 1],
                                l_pos_dic[i - e[0] + sen_len], l_pos_dic[i - e[1] + sen_len]))
            SF_one.append(_)
            if isPCNN:
                if i == e[0] + 1:
                    while len(SF_one) < S1:
                        SF_one.append(t_PAD)
                elif i == e[1] + 2:
                    while len(SF_one) < S1 + S2:
                        SF_one.append(t_PAD)
                elif all(sen[i] == F_END):
                    while len(SF_one) < S1 + S2 + S3:
                        SF_one.append(t_PAD)
                    break
        WLF.append(np.concatenate(
            (sen[e[0]], sen[e[0] + 1], sen[e[0] + 2], sen[e[1]], sen[e[1] + 1], sen[e[1] + 2])))
        SF.append(np.asarray(SF_one))
    SF_r, WLF = np.stack(SF), np.asarray(WLF)

    return SF_r, WLF


def get_batches(xdata, ydata, batch_size, num_epochs, shuffle=True):
    """
    生成一个batch迭代器
    """
    data_size = len(xdata)
    num_batches_per_epoch = int((len(xdata) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 混洗
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(data_size)
            xdata = xdata[shuffle_indices]
            ydata = ydata[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield xdata[start_index:end_index], ydata[start_index:end_index]
