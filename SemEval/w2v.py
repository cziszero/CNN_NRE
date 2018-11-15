# coding: utf-8
from constant import *
import numpy as np
import pickle
import gensim
from gensim.models import Word2Vec
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import metrics
import csv
import os
import time
from m_utils import *

# find_hp_leap()
# all_sens = 'all_sens.pk'  # Word2Vec训练所用语料库


def find_s2p(his):
    name_hps = ['sg', 'hs', 'size', 'window', 'm_iter']
    ch_hps = [ch_sg, ch_hs, ch_size, ch_window, ch_m_iter]
    t_hps = {'sg': 0, 'hs': 0, 'size': 300, 'window': 4, 'm_iter': 500}
    score_dict = {}
    for i in range(len(his)):
        sc = his[i]
        t_ind = i % len(name_hps)
        t_name = name_hps[t_ind]
        # print('test for %s' % (t_name))
        for j in range(len(ch_hps[t_ind])):
            pe = ch_hps[t_ind][j]
            t_hps[t_name] = pe
            # print(t_hps, ' ', sc[j])
            if str(t_hps) not in score_dict:
                score_dict[get_dict_key(t_hps)] = [sc[j]]
            else:
                score_dict[get_dict_key(t_hps)].append(sc[j])
    f = open('score_dict.txt', 'w')
    for k, v in score_dict.items():
        print("%s %d %f" % (k, len(v), max(v) - min(v)))
        f.write("%s %s\n" % (k, str(v)))
    f.close()
    return score_dict


def train_one_dict(sens, size=100, sg=0, hs=0, m_iter=5, window=5,
                   senfilename='train_word2vec.txt', outfilename=None,
                   save=True, fromFile=False):
    """
    通过查询已训练的或重新训练一个Word2Vec模型。
    """
    class MySentences(object):

        def __init__(self, fileName):
            self.fileName = fileName

        def __iter__(self):
            for line in open(self.fileName, 'r', encoding='utf-8'):
                yield line.split()
    if os.path.exists(outfilename):
        print('load file %s' % (outfilename))
        return Word2Vec.load(outfilename)  # 如果已经有了则直接载入返回
    if fromFile:
        sens = MySentences(senfilename)  # 训练数据在文件中
    elif sens == None:
        sens = pickle.load(open(all_sens, 'rb'))  # 已处理好的预料存在all_sens中
    print("train %s" % (outfilename))
    model = Word2Vec(sens, min_count=1, window=window,
                     size=size, sg=sg, hs=hs, iter=m_iter, workers=8)
    if save and outfilename != None:
        model.save(outfilename)
        print("save Word2Vec model to %s" % (outfilename))
    return model


def train_diff_dict():
    sens = pickle.load(open(all_sens, 'rb'))
    for sg in ch_sg:
        for hs in ch_hs:
            for size in ch_size:
                for window in ch_window:
                    for m_iter in ch_m_iter:
                        outfilename = w2vfileformat % (
                            window, size, m_iter, sg, hs)
                        print(outfilename)
                        train_one_dict(sens, size=size, sg=sg,
                                       hs=hs, m_iter=m_iter, window=window, outfilename=outfilename, save=True)


def make_net(class_num, fea_size, optimizer='sgd'):
    model = Sequential()
    la = Dense(units=class_num, input_shape=(fea_size,), activation='softmax')
    model.add(la)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def test_word2vec_one(w2v=None, w2v_g=None, fea_size=100, class_num=19):
    reader = csv.reader(open('data/entitys_%d.txt' %
                             (class_num), 'r', encoding='utf-8'))
    fea = []
    if w2v_g == None:
        for i in reader:
            fea.append(w2v[i[0]][0:fea_size] - w2v[i[1]][0:fea_size])
    else:
        for i in reader:
            try:
                fea.append(w2v_g[i[0]][0:fea_size] - w2v_g[i[1]][0:fea_size])
            except KeyError:
                fea.append(w2v[i[0]][0:fea_size] - w2v[i[1]][0:fea_size])

    train_y = np.load('data/train_y_%d.npy' % (class_num))
    test_y = np.load('data/test_y_%d.npy' % (class_num))
    train_x = np.asarray(fea[0:len(train_y)])
    test_x = np.asarray(fea[len(train_y):len(fea)])

    # 按照class_num类训练和预测
    # K.clear_session()
    model = make_net(class_num, fea_size)
    es = EarlyStopping(monitor='loss', patience=10)
    his = model.fit(train_x, train_y, epochs=3000, callbacks=[es], verbose=0)
    pred = model.predict(test_x)
    real_lab_num, pred_lab_num, pt_lab_num, acc, recall, f, f_mean = calc_F(
        test_y, pred, class_num=class_num)

    return his, f_mean


def test_random_w2v():
    w2v = {}
    class_num = 18
    with open('data/entitys_%d.txt' % (class_num), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i in reader:
            w2v[i[0]] = np.random.normal(size=(300,))
            w2v[i[1]] = np.random.normal(size=(300,))
    for si in ch_size:
        _, f = test_word2vec_one(
            w2v=w2v, w2v_g=None, fea_size=si, class_num=class_num)
        print(si, ' ', f)

test_random_w2v()


def test_word2vec():
    f = []
    hps = {}
    class_num = 19
    for sg in ch_sg:
        for hs in ch_hs:
            for size in ch_size:
                for window in ch_window:
                    for m_iter in ch_m_iter:
                        hps['window'], hps['size'], hps['m_iter'], hps[
                            'sg'], hps['hs'] = window, size, m_iter, sg, hs
                        t_f = get_one_res(
                            hps=hps, class_num=class_num, w2v_g=None)


def get_one_res(hps, class_num, w2v_g=None):
    score_dict = pickle.load(open('score_dict_%d.pk' % (class_num), 'rb'))
    dk = get_dict_key(hps)
    print(dk)
    if score_dict.get(dk):
        t_f = score_dict.get(dk)
    else:
        get_time(str(hps))
        ofn = getOutfilename(hps)
        w2v = train_one_dict(sens=None, size=hps['size'], sg=hps['sg'], hs=hps['hs'], m_iter=hps[
            'm_iter'], window=hps['window'], outfilename=ofn, save=True, fromFile=False)
        _, t_f = test_word2vec_one(w2v=w2v, w2v_g=w2v_g, fea_size=hps[
            'size'], class_num=class_num)
        score_dict[dk] = t_f
        pickle.dump(score_dict, open('score_dict_%d.pk' % (class_num), 'wb'))
    return t_f


def find_hp_leap_one(name, ch, hps, class_num, w2v_g=None):
    f, best = [], 0
    for i in ch:
        hps[name] = i
        t_f = get_one_res(hps=hps, class_num=class_num, w2v_g=w2v_g)
        print('find_hp_leap_one', hps, t_f)
        f.append(t_f)
    best = max(f)
    f = np.asarray(f)
    hps[name] = ch[np.argmax(f)]
    return hps.copy(), f, best


def test_google_w2v(hps={'sg': 0, 'hs': 0, 'size': 150, 'window': 3, 'm_iter': 500}):
    get_time('load GoogleNews-vectors-negative300.bin')
    w2v_g = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)
    get_time('load GoogleNews-vectors-negative300.bin successful')
    hps, f, best = find_hp_leap_one(
        name='size', ch=ch_size, hps=hps, class_num=19, w2v_g=w2v_g)
    pickle.dump(hps, open('google_hps.pk', 'wb'))
    pickle.dump(f, open('google_f.pk', 'wb'))
    pickle.dump(best, open('google_best.pk', 'wb'))
    return


def find_hp_leap(name_hps=['sg', 'hs', 'size', 'window', 'm_iter'],
                 ch_hps=[ch_sg, ch_hs, ch_size, ch_window, ch_m_iter],
                 t_hps={'sg': 0, 'hs': 0, 'size': 300, 'window': 4, 'm_iter': 500}):
    """
    parameter：
    name_hps:要寻找的超参数名
    ch_hps:每种超参数的候选值
    t_hps:起始值
    return hps，best，his
    hps:与best对应的参数值
    best:对每类候选值的测试
    his:所有测试值的记录
    """
    hps, his, best = [], [], []
    flag = 0
    class_num = 19

    get_time('start find_hp_leap')
    while flag != len(name_hps):
        flag = 0
        for name, ch in zip(name_hps, ch_hps):
            t_hps, t_his, t_best = find_hp_leap_one(
                name=name, ch=ch, hps=t_hps, class_num=class_num)
            get_time("%s %s %f \n" % (name, str(t_hps), t_best))
            his.append(t_his)
            best.append(t_best)
            hps.append(t_hps)
            if t_best - best[len(best) - 2] < leap_eps:
                flag += 1
            pickle.dump(hps, open('hps.pk', 'wb'))
            pickle.dump(best, open('best.pk', 'wb'))
            pickle.dump(his, open('his.pk', 'wb'))

    return hps, best, his


def ttt(w2v):
    reader = csv.reader(open('entitys.txt', 'r', encoding='utf-8'))
    writer = csv.writer(open('exist.txt', 'w', encoding='utf-8'))
    for i in reader:
        try:
            a, b = w2v[i[0]], w2v[i[1]]
            writer.writerow(i)
        except KeyError:
            try:
                a = lemmatizer.lemmatize(i[0])
                b = lemmatizer.lemmatize(i[1])
                a, b = w2v[a], w2v[b]
                writer.writerow(i)
            except KeyError:
                print(i)
