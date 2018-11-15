import numpy as np
import pickle

eps = 0.0001
leap_eps = 0.0001

ch_sg = [0, 1]
cs_sg = len(ch_sg)
ch_hs = [0, 1]
cs_hs = len(ch_hs)
ch_size = [100, 150, 200, 250, 300]
cs_size = len(ch_size)
ch_window = [3, 4, 5, 6, 7, 8, 9, 10]
cs_window = len(ch_window)
ch_m_iter = [5, 50, 500, 1000, 2000, 3000]
cs_m_iter = len(ch_m_iter)
ch_epoch = [10, 50, 100, 200, 300, 400, 500, 1000]
cs_epoch = len(ch_epoch)

w2vfileformat = 'train_SemEval_word2vec_w_%d_size_%d_iter_%d_sg_%d_hs_%d.bin'

window_size = 3  # 窗口大小
PF_size = 5  # PF的维度
WF_size = 50  # 词向量维度
sen_len = 90  # 句子的最大长度
class_num = 19  # 一共有多少类
filter_hight = 3  # 卷积核高度
n1 = 200  # 隐藏层1的神经元数目，也即卷积核个数
n2 = 100  # 隐藏层1的神经元数目，即全连接层的神经元个数
batch_size = 160  # 每一批的大小
lr = 0.001  # 学习速率
dropout = 0.5  # Dropout比率
num_epochs = 3000  # 训练轮次
evaluate_every = 100  # 每训练多少轮次测试评估一下
K = 4  # k折交叉验证
S1 = 67
S2 = 34
S3 = 73

# 填充和初始化
F_START = np.ones(shape=WF_size, dtype=float) / 2
F_E1 = np.ones(shape=WF_size, dtype=float) / 4
F_E2 = np.ones(shape=WF_size, dtype=float) / 8
F_END = np.ones(shape=WF_size, dtype=float)
F_PAD = np.zeros(shape=WF_size, dtype=float)
l_pos_dic = pickle.load(open('data/l_pos_dic.pk', 'rb'))
r_pos_dic = pickle.load(open('data/r_pos_dic.pk', 'rb'))

# 单词形式
fn_sens_all = 'data/sens_all.pk'
fn_sens_test = 'data/sens_test.pk'
fn_sens_train = 'data/sens_train.pk'
fn_pos_all = 'data/pos_all.pk'
fn_pos_test = 'data/pos_test.pk'
fn_pos_train = 'data/pos_train.pk'
fn_test_file = 'data/test_file.txt'
fn_train_file = 'data/train_file.txt'
fn_all_file = 'data/all.txt'

# 标定数据
fn_train_y = 'data/train_y_%d.npy' % (class_num)
fn_test_y = 'data/test_y_%d.npy' % (class_num)

# 输入
fn_words_vec_base = 'data/words_vec_%s.pk'
fn_sens_vec_all_base = 'data/sens_vec_all_%s.pk'
fn_sens_vec_train_base = 'data/sens_vec_train_%s.pk'
fn_sens_vec_test_base = 'data/sens_vec_test_%s.pk'

fn_cnn_his_base = 'result/cnn_%s_his_%s_%s.pk'
fn_cnn_pred_base = 'result/cnn_%s_pred_%s_%s.pk'
fn_cnn_model_base = 'model/cnn_%s_%s_%s.h5'

fn_SF_train_base = 'data/SF_train_%s_%s.npy'
fn_WLF_train_base = 'data/WLF_train_%s_%s.npy'
fn_SF_test_base = 'data/SF_test_%s_%s.npy'
fn_WLF_test_base = 'data/WLF_test_%s_%s.npy'
