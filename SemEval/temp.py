from preRawFile import *
from data_prepare import *
from constant import *
from m_utils import *

import pickle as pk
import numpy as np

from keras import backend as K
from keras.models import Sequential
import keras.layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import metrics

import os

num_epochs = 3000  # 训练轮次
evaluate_every = 100  # 每训练多少轮次测试评估一下
K = 4  # k折交叉验证
S1 = 67
S2 = 34
S3 = 73

dropout = 0.5
withWF = False
optimizer = 'sgd'
inputs = Input(shape=(S1 + S2 + S3, 3 * WF_size + 2 * PF_size))
conv = Conv1D(filters=n1, kernel_size=(
    filter_hight), padding="same")(inputs)
# a = conv[:, 0:S1, :]
# b = conv[:, S1:S1 + S2, :]
# c = conv[:, S2:S1 + S2 + S3, :]
a = Lambda(lambda x: x[:, 0:S1, :])(conv)
b = Lambda(lambda x: x[:, S1:S1 + S2, :])(conv)
c = Lambda(lambda x: x[:, S1 + S2:S1 + S2 + S3, :])(conv)
maxa = Flatten()(MaxPool1D(S1)(a))
maxb = Flatten()(MaxPool1D(S2)(b))
maxc = Flatten()(MaxPool1D(S3)(c))
lmax = keras.layers.concatenate([maxa, maxb, maxc])
lda = Dense(units=n2, activation='tanh')(lmax)
if not withWF:
    lda = Dropout(dropout)(lda)
    predictions = Dense(units=class_num, activation='softmax')(lda)
    model = Model(inputs=inputs, outputs=predictions)
else:
    wf = Input(shape=(window_size * WF_size * 2,))
    x = keras.layers.concatenate([lda, wf])
    x = Dropout(dropout)(x)
    predictions = Dense(units=class_num, activation='softmax')(x)
    model = Model(inputs=[inputs, wf], outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
