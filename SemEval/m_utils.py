import time
import numpy as np
from constant import *


def get_dict_key(hps):
    return "{'size': %d, 'sg': %d, 'm_iter': %d, 'window': %d, 'hs': %d}" % (hps['size'], hps['sg'], hps['m_iter'], hps['window'], hps['hs'])


def getOutfilename(hps):
    return w2vfileformat % (hps['window'], hps['size'], hps['m_iter'], hps['sg'], hps['hs'])


def get_time(st):
    print(st)
    print(time.strftime(" %a %b %d %H:%M:%S %Y", time.localtime()))
    return time.time()


def load_leap():
    """
    return best, his, hps
    """
    import pickle as pk
    best = pk.load(open('result/leap_best.pk', 'rb'))
    his = pk.load(open('result/leap_his.pk', 'rb'))
    hps = pk.load(open('result/leap_hps.pk', 'rb'))
    return best, his, hps


def calc_F(y_real, y_pred, class_num):
    class_num = 19
    real_lab = y_real.argmax(1)
    pred_lab = y_pred.argmax(1)
    real_lab_num = []
    pred_lab_num = []
    pt_lab_num = []
    acc, recall, f = [], [], []
    for i in range(class_num):
        r = (real_lab == i)
        p = (pred_lab == i)
        rp = np.stack((r, p))
        real_lab_num.append(np.sum(r))
        pred_lab_num.append(np.sum(p))
        pt_lab_num.append(np.sum(rp.all(0)))
        if pred_lab_num[i] > eps:
            acc.append(pt_lab_num[i] / pred_lab_num[i])
        else:
            acc.append(0)
        if real_lab_num[i] > eps:
            recall.append(pt_lab_num[i] / real_lab_num[i])
        else:
            recall.append(0)
        if acc[i] + recall[i] > eps:
            f.append(2 * acc[i] * recall[i] / (acc[i] + recall[i]))
        else:
            f.append(0)

    real_lab_num = np.asarray(real_lab_num)
    pred_lab_num = np.asarray(pred_lab_num)
    pt_lab_num = np.asarray(pt_lab_num)
    acc = np.nan_to_num(np.asarray(acc))
    recall = np.asarray(recall)
    f = np.nan_to_num(np.asarray(f))
    # s = 2 * acc[0] * recall[0] / (acc[0] + recall[0]) # 计算Other类
    s = 0  # 不计算Other类
    for i in range(1, 19, 2):
        a = pt_lab_num[i] + pt_lab_num[i + 1]
        b = pred_lab_num[i] + pred_lab_num[i + 1]
        c = real_lab_num[i] + real_lab_num[i + 1]
        if b > eps:
            ac = a / b
        else:
            ac = 0
        if c > eps:
            re = a / c
        else:
            re = 0
        if ac + re > eps:
            s += (2 * ac * re) / (ac + re)
    return real_lab_num, pred_lab_num, pt_lab_num, acc, recall, f, s / 9
