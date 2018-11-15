from pre4word2vec import *
#from m_utils import *
#from analysis import *
#from preRawFile import *

#t0 = get_time("start ")
# train_diff_dict()
# t1 = get_time("finish w2v,start classify")
# test_word2vec()
# t2 = get_time("finish classify")

# test_word2vec_one(fea_size=100, epoch_num=50,
#   filename='train_SemEval_word2vec_w_4_size_100_iter_3000_sg_0_hs_0.bin')

# w2v = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


# train_y = np.load('train_set_y.npy')
# test_y = np.load('test_set_y.npy')
# ind = train_y.argmax(1) > 0
# train_y = train_y[ind, 1:19]
# ind = test_y.argmax(1) > 0
# test_y = test_y[ind, 1:19]
# find_hp_leap()
# test_google_w2v()
# find_hp_leap()

# best, his, hps = load_leap()
# find_s2p(his)

# hps, best, his = find_hp_leap(
#     t_hps={'sg': 0, 'hs': 0, 'size': 100, 'window': 3, 'm_iter': 10})


test_word2vec()
