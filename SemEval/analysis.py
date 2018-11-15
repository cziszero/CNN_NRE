import pickle as pk
import numpy as np
from constant import *
from m_utils import calc_F
from m_utils import get_dict_key
import matplotlib.pyplot as plt


def w2vvsrandom():
    ran = [0.155377468047, 0.202006710659,
           0.216982192735, 0.229992878543, 0.238050091863]
    a = pk.load(open('result/score_dict_18.pk', 'rb'))
    w2v = [0, 0, 0, 0, 0]
    for i in a:
        for j in range(len(ch_size)):
            if "'size': %d" % ch_size[j] in i:
                w2v[j] = max(w2v[j], a[i])
    print(w2v)
    n_groups = 5
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    rects1 = plt.bar(index, ran, bar_width, alpha=opacity, label='Random')
    rects2 = plt.bar(index + bar_width, w2v, bar_width,
                     alpha=opacity, label='Word2Vec')

    plt.xlabel('Word Embedding Size')
    plt.ylabel('Macro F1')
    # plt.title('Scores by group and gender')
    plt.xticks(index + bar_width, ch_size)
    # plt.ylim(0, 40)
    plt.legend()

    plt.tight_layout()
    plt.show()

w2vvsrandom()


def getres():
    fn = ["cnn_SF_pred_random_cnn.pk",    "cnn_SF_pred_random_pcnn.pk",
          "cnn_SF_pred_w2v_cnn.pk",       "cnn_SF_pred_w2v_pcnn.pk",
          "cnn_SFWF_pred_random_cnn.pk",  "cnn_SFWF_pred_w2v_pcnn.pk",
          "cnn_SFWF_pred_random_pcnn.pk", "cnn_SFWF_pred_w2v_cnn.pk"]
    test_y = np.load(fn_test_y)
    for i in fn:
        pred = pk.load(open('result/' + i, 'rb'))
        real_lab_num, pred_lab_num, pt_lab_num, acc, recall, f, f_mean = calc_F(
            test_y, pred, class_num=19)
        print(i, f_mean)
    return


def union_score():
    a = pk.load(open('result/score_dict_19.pk', 'rb'))
    b = pk.load(open('result/grid.pk', 'rb'))
    hps = {}
    class_num = 18
    ind = -1
    for sg in ch_sg:
        for hs in ch_hs:
            for size in ch_size:
                for window in ch_window:
                    for m_iter in ch_m_iter:
                        ind += 1
                        if ind == len(b):
                            return a
                        hps['window'], hps['size'], hps['m_iter'], hps[
                            'sg'], hps['hs'] = window, size, m_iter, sg, hs
                        key = get_dict_key(hps)

                        if b[ind][0] > 0 and a.get(key, 0) == 0:
                            print(key, b[ind][0])
                            a[key] = b[ind][0]

                        # a[key] = max(a.get(key,0),)


def find_hp_leap_one_t(name, ch, hps, sd, hh):
    f, best = [], 0
    for i in ch:
        hps[name] = i
        t_f = sd.get(get_dict_key(hps), 0)
        hh[get_dict_key(hps)] = 1
        if t_f == 0:
            print('*' * 10)
        print('find_hp_leap_one', hps, t_f)
        f.append(t_f)
    best = max(f)
    f = np.asarray(f)
    hps[name] = ch[np.argmax(f)]
    return hps.copy(), f, best


def getleap():
    name_hps = ['sg', 'hs', 'size', 'window', 'm_iter']
    ch_hps = [ch_sg, ch_hs, ch_size, ch_window, ch_m_iter]
    t_hps = {'sg': 0, 'hs': 0, 'size': 100, 'window': 3, 'm_iter': 5}
    sd = pk.load(open('result/score_dict_18.pk', 'rb'))
    hps, his, best = [], [], []
    flag = 0
    class_num = 19
    hh = {}
    while flag != len(name_hps):
        flag = 0
        for name, ch in zip(name_hps, ch_hps):
            t_hps, t_his, t_best = find_hp_leap_one_t(
                name=name, ch=ch, hps=t_hps,  sd=sd, hh=hh)
            his.append(t_his)
            best.append(t_best)
            hps.append(t_hps)
            if t_best - best[len(best) - 2] < leap_eps:
                flag += 1
    print('all setting num', len(hh))
    plt.ylabel('Macro F1')
    plt.xlabel('epoch')
    plt.plot(best, '-o')
    plt.savefig('leap.png')
    return hps, best, his

# getres()
# hps, best, his = getleap()
# ua = union_score()


def getInd(ch):
    ind = np.arange(max(ch) + 1)
    for i in range(len(ch)):
        ind[ch[i]] = i
    return ind


def vis(f):
    i_sg = getInd(ch_sg)
    i_hs = getInd(ch_hs)
    i_size = getInd(ch_size)
    i_window = getInd(ch_window)
    i_m_iter = getInd(ch_m_iter)
    i_epoch = getInd(ch_epoch)
    ind = -1
    scores = np.zeros((cs_sg, cs_hs, cs_size, cs_window, cs_m_iter, cs_epoch))
    for sg in ch_sg:
        for hs in ch_hs:
            for size in ch_size:
                for window in ch_window:
                    for m_iter in ch_m_iter:
                        for epoch_num in ch_epoch:
                            ind += 1
                            if (ind >= len(f)):
                                return scores
                            if f[ind] == max(f):
                                print("%f %d %d %d %d %d %d\n" % (
                                    f[ind], sg, hs, size, window, m_iter, epoch_num))
                            scores[i_sg[sg], i_hs[hs], i_size[size], i_window[
                                window], i_m_iter[m_iter], i_epoch[epoch_num]] = f[ind]

# f = pickle.load(open('f.pk', 'rb'))
# f = np.asarray(f)
# scores_19 = vis(f[:, 0])
# scores_18 = vis(f[:, 1])
