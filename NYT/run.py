import tensorflow as tf
from train import train
from test import test
from network import Settings
import numpy as np
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper',
                           'the user you want to send info to')

itchat_run = True
if itchat_run:
    import itchat
    itchat.auto_login(hotReload=True)


def get_time(st):
    try:
        time.time()
    except NameError:
        import time
    finally:
        print(str(st) + time.strftime(" %a %b %d %H:%M:%S %Y", time.localtime()))
        return time.time()


MAXMode_normal = 'MAX_Normal'  # 普通的Max-Pooling
MAXMode_TopK = 'MAX_TopK'  # 'TopK' # K-Max-Pooling
ATTMode_one = 'ATT_One'  # 每个bag中只取一个
ATTMode_TopK = 'ATT_opK'  # 每个bag中取前K个
ATTMode_Weight = 'ATT_Weight'  # 每个bag中按权值加和
ATTMode_Avg = 'ATT_Avg'  # 每个bag中句子取平均


def run(settings, debug=True):
    if not debug:
        ts = "train_%s %s" % (settings.MAXMode, settings.ATTMode)
        t0 = get_time('%s start' % ts)
        try:
            train(settings, wordembedding, itchat_run, FLAGS)
        except BaseException as be:
            print('%s failed' % ts)
            print(be)
            return
        t1 = get_time("%s end " % ts)
        print("%s spend %f minutes" % (ts, (t1 - t0) / 60))
        ts = "test_%s %s" % (settings.MAXMode, settings.ATTMode)
        t0 = get_time('%s start' % ts)
        try:
            test(settings, wordembedding, itchat_run, FLAGS)
        except BaseException as be:
            print('%s failed' % ts)
            print(be)
            return
        t1 = get_time("%s end " % ts)
        print("%s spend %f minutes" % (ts, (t1 - t0) / 60))
    else:
        ts = "train_%s %s" % (settings.MAXMode, settings.ATTMode)
        t0 = get_time('%s start' % ts)
        train(settings, wordembedding, itchat_run, FLAGS, t='_sim')
        t1 = get_time("%s end " % ts)
        print("%s spend %f minutes" % (ts, (t1 - t0) / 60))
        ts = "test_%s %s" % (settings.MAXMode, settings.ATTMode)
        t0 = get_time('%s start' % ts)
        test(settings, wordembedding, itchat_run, FLAGS)
        t1 = get_time("%s end " % ts)
        print("%s spend %f minutes" % (ts, (t1 - t0) / 60))


print('reading wordembedding')
mode = bool(sys.argv[1])

wordembedding = np.load('./data/vec.npy')
print(mode)
settings = Settings(MAXMode=MAXMode_normal, ATTMode=ATTMode_Weight)
run(settings, debug=mode)
settings = Settings(MAXMode=MAXMode_normal, ATTMode=ATTMode_one)
run(settings, debug=mode)
settings = Settings(MAXMode=MAXMode_TopK, ATTMode=ATTMode_Weight)
run(settings, debug=mode)
settings = Settings(MAXMode=MAXMode_TopK, ATTMode=ATTMode_one)
run(settings, debug=mode)
