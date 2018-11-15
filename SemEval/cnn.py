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
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import metrics

import os


def make_cnn_net(dropout=0.5, withWF=False, optimizer='sgd'):
    inputs = Input(shape=(sen_len - 2, 3 * WF_size + 2 * PF_size))
    conv = Conv1D(filters=n1, kernel_size=(
        filter_hight), padding="same")(inputs)
    lmax = MaxPool1D(sen_len - 2)(conv)
    lmax = Flatten()(lmax)
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
    return model


def make_pcnn_net(dropout=0.5, withWF=False, optimizer='sgd'):
    inputs = Input(shape=(S1 + S2 + S3, 3 * WF_size + 2 * PF_size))
    conv = Conv1D(filters=n1, kernel_size=(
        filter_hight), padding="same")(inputs)
    a = Lambda(lambda x: x[:, 0:S1, :])(conv)
    b = Lambda(lambda x: x[:, S1:S1 + S2, :])(conv)
    c = Lambda(lambda x: x[:, S1 + S2:S1 + S2 + S3, :])(conv)
    maxa = Flatten()(MaxPool1D(S1)(a))
    maxb = Flatten()(MaxPool1D(S2)(b))
    maxc = Flatten()(MaxPool1D(S3)(c))
    lmax = x = keras.layers.concatenate([maxa, maxb, maxc])
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

    return model


def test_cnn(isRandom, isPCNN):
    if not isRandom:
        t = 'w2v'
    else:
        t = 'random'
    if not isPCNN:
        tb = 'cnn'
    else:
        tb = 'pcnn'
    fn_words_vec = fn_words_vec_base % (t)
    fn_sens_vec_all = fn_sens_vec_all_base % (t)
    fn_sens_vec_train = fn_sens_vec_train_base % (t)
    fn_sens_vec_test = fn_sens_vec_test_base % (t)

    fn_SF_train = fn_SF_train_base % (t, tb)
    fn_WLF_train = fn_WLF_train_base % (t, tb)
    fn_SF_test = fn_SF_test_base % (t, tb)
    fn_WLF_test = fn_WLF_test_base % (t, tb)
    print(fn_words_vec, fn_SF_train, fn_WLF_train, fn_SF_test,
          fn_WLF_test, fn_sens_vec_all, fn_sens_vec_train, fn_sens_vec_test)
    if not os.path.exists(fn_words_vec):
        get_time('mk sens vec() to make word vec dict---start')
        sens = pk.load(open(fn_sens_all, 'rb'))
        mk_sens_vec(isRandom, sens, fn_sens_vec_all,
                    fn_words_vec)  # 构造一个词典，便于之后查找，分为w2v词典和random词典
        get_time('mk sens vec() to make word vec dict---end')

    if not os.path.exists(fn_SF_test):
        get_time('get_sens_vec() to make input data---start')
        SF_train, WLF_train = get_sens_vec(isPCNN, isRandom,
                                           pk.load(open(fn_sens_train, 'rb')), pk.load(
                                               open(fn_pos_train, 'rb')),
                                           fn_words_vec, fn_sens_vec_train)
        SF_test, WLF_test = get_sens_vec(isPCNN, isRandom,
                                         pk.load(open(fn_sens_test, 'rb')), pk.load(
                                             open(fn_pos_test, 'rb')),
                                         fn_words_vec, fn_sens_vec_test)
        np.save(fn_SF_test, SF_test)
        np.save(fn_SF_train, SF_train)
        np.save(fn_WLF_test, WLF_test)
        np.save(fn_WLF_train, WLF_train)
        get_time('get_sens_vec() to make input data---end')
    else:
        get_time('load input data')
        SF_test = np.load(fn_SF_test)
        SF_train = np.load(fn_SF_train)
        WLF_test = np.load(fn_WLF_test)
        WLF_train = np.load(fn_WLF_train)

    def fit_test(withWF, fn_his, fn_pred, fn_model, isPCNN):
        es = EarlyStopping(monitor='loss', patience=10)
        get_time('输入数据准备完成，开始训练')
        if not isPCNN:
            model = make_cnn_net(withWF=withWF)
        else:
            model = make_pcnn_net(withWF=withWF)
        if not withWF:
            his = model.fit(SF_train, train_y, epochs=num_epochs,
                            callbacks=[es], verbose=1, batch_size=batch_size)
        else:
            his = model.fit([SF_train, WLF_train], train_y,
                            epochs=num_epochs, callbacks=[es], verbose=1, batch_size=batch_size)
        model.save(fn_model)
        get_time('训练完成，开始测试，模型保存在%s' % (fn_model))
        if not withWF:
            pred = model.predict(SF_test)
        else:
            pred = model.predict([SF_test, WLF_test])
        real_lab_num, pred_lab_num, pt_lab_num, acc, recall, f, f_mean = calc_F(
            test_y, pred, class_num=class_num)
        try:
            pk.dump(his.history, open(fn_his, 'wb'))
            pk.dump(pred, open(fn_pred, 'wb'))
        except:
            pass
        get_time('测试完成')
        print('F1值为', f_mean)
        return f_mean

    get_time('cnn %s %s SF satrt' % (t, tb))
    fn_cnn_his, fn_cnn_pred, fn_cnn_model = fn_cnn_his_base % (
        'SF', t, tb), fn_cnn_pred_base % ('SF', t, tb), fn_cnn_model_base % ('SF', t, tb)
    f1_sf = fit_test(withWF=False, fn_his=fn_cnn_his,
                     fn_pred=fn_cnn_pred, fn_model=fn_cnn_model, isPCNN=isPCNN)
    get_time('cnn %s %s SF end' % (t, tb))

    get_time('cnn %s %s SFWF start' % (t, tb))
    fn_cnn_his, fn_cnn_pred, fn_cnn_model = fn_cnn_his_base % (
        'SFWF', t, tb), fn_cnn_pred_base % ('SFWF', t, tb), fn_cnn_model_base % ('SFWF', t, tb)
    f1_sfwf = fit_test(withWF=True,  fn_his=fn_cnn_his,
                       fn_pred=fn_cnn_pred, fn_model=fn_cnn_model, isPCNN=isPCNN)
    get_time('cnn %s %s SFWF end' % (t, tb))
    return f1_sf, f1_sfwf


if __name__ == '__main__':
    if not os.path.exists(fn_sens_all):
        get_time('read_data to make raw sens---start')
        set_size, _, (s1, s2, s3), sens, pos, rs = read_data(fn_all_file)
        pk.dump(sens, open(fn_sens_all, 'wb'))
        pk.dump(pos, open(fn_pos_all, 'wb'))
        set_size, _, (s1, s2, s3), sens, pos, rs = read_data(fn_test_file)
        pk.dump(sens, open(fn_sens_test, 'wb'))
        pk.dump(pos, open(fn_pos_test, 'wb'))
        set_size, _, (s1, s2, s3), sens, pos, rs = read_data(fn_train_file)
        pk.dump(sens, open(fn_sens_train, 'wb'))
        pk.dump(pos, open(fn_pos_train, 'wb'))
        get_time('read_data to make raw sens---end')

    # set_size, _, (s1, s2, s3), sens, pos, rs = read_data(fn_all_file)
    # print(s1, s2, s3)
    train_y = np.load(fn_train_y)
    test_y = np.load(fn_test_y)

    # # 使用Word2Vec初始化 PCNN结构
    # st = '使用Word2Vec初始化 PCNN结构'
    # get_time('%s start' % (st))
    # f1_sf, f1_sfwf = test_cnn(isRandom=False, isPCNN=True)
    # get_time('%s f1_sf %f f1_sf_wf %f' % (st, f1_sf, f1_sfwf))

    # # 使用Random初始化 PCNN结构
    # st = '使用Random初始化 PCNN结构'
    # get_time('%s start' % (st))
    # f1_sf, f1_sfwf = test_cnn(isRandom=True, isPCNN=True)
    # get_time('%s f1_sf %f f1_sf_wf %f' % (st, f1_sf, f1_sfwf))

    # 使用Word2Vec初始化 CNN结构
    st = '使用Word2Vec初始化 CNN结构'
    get_time('%s start' % (st))
    f1_sf, f1_sfwf = test_cnn(isRandom=False, isPCNN=False)
    get_time('%s f1_sf %f f1_sf_wf %f' % (st, f1_sf, f1_sfwf))

    # 使用Random初始化 CNN结构
    st = '使用Random初始化 CNN结构'
    get_time('%s start' % (st))
    f1_sf, f1_sfwf = test_cnn(isRandom=True, isPCNN=False)
    get_time('%s f1_sf %f f1_sf_wf %f' % (st, f1_sf, f1_sfwf))
