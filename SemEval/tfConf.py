# coding=utf-8

import tensorflow as tf
# 声明可使用几号显卡，可选 0 1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getSess():
    # 显存按需分配
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess