import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score

pathname = "./model/CNN_model-"

wordembedding = np.load('./data/vec.npy')

test_settings = network.Settings()
test_settings.vocab_size = len(wordembedding)
test_settings.bag_num = 262 * 9

bag_num_test = test_settings.bag_num
sess = tf.Session()
with tf.variable_scope("model"):
    mtest = network.CNN(is_training=False,
                        word_embeddings=wordembedding,
                        settings=test_settings)

model_iter = 17000
saver = tf.train.Saver()
saver.restore(sess, pathname + str(model_iter))


def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

    feed_dict = {}
    total_shape = []
    total_num = 0
    total_word = []
    total_pos1 = []
    total_pos2 = []

    for i in range(len(word_batch)):
        total_shape.append(total_num)
        total_num += len(word_batch[i])
        for word in word_batch[i]:
            total_word.append(word)
        for pos1 in pos1_batch[i]:
            total_pos1.append(pos1)
        for pos2 in pos2_batch[i]:
            total_pos2.append(pos2)

    total_shape.append(total_num)
    total_shape = np.array(total_shape)
    total_word = np.array(total_word)
    total_pos1 = np.array(total_pos1)
    total_pos2 = np.array(total_pos2)

    feed_dict[mtest.total_shape] = total_shape
    feed_dict[mtest.input_word] = total_word
    feed_dict[mtest.input_pos1] = total_pos1
    feed_dict[mtest.input_pos2] = total_pos2
    feed_dict[mtest.input_y] = y_batch

    loss, accuracy, prob = sess.run(
        [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
    return prob, accuracy


test_y = np.load('./data/testall_y.npy')
test_word = np.load('./data/testall_word.npy')
test_pos1 = np.load('./data/testall_pos1.npy')
test_pos2 = np.load('./data/testall_pos2.npy')
allprob = []
acc = []
for i in range(int(len(test_word) / float(test_settings.bag_num))):
    m_st, m_ed = i * test_settings.bag_num, (i + 1) * test_settings.bag_num
    prob, accuracy = test_step(test_word[m_st:m_ed],
                               test_pos1[m_st:m_ed],
                               test_pos2[m_st:m_ed],
                               test_y[m_st:m_ed])
    m_ta = np.reshape(np.array(accuracy), (test_settings.bag_num))
    m_acc = np.mean(m_ta)
    acc.append(m_acc)

    prob = np.reshape(np.array(prob), (test_settings.bag_num,
                                       test_settings.num_classes))
    for single_prob in prob:
        allprob.append(single_prob[1:])
allprob = np.reshape(np.array(allprob), (-1))
order = np.argsort(-allprob)

print('saving all test result...')
current_step = model_iter

# ATTENTION: change the save path before you save your result !!
np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
allans = np.load('./data/allans.npy')

# caculate the pr curve area
average_precision = average_precision_score(allans, allprob)
print('PR curve area:' + str(average_precision))

time_str = datetime.datetime.now().isoformat()
print(time_str)
print('P@N for all test data:')
print('P@100:')
top100 = order[:100]
correct_num_100 = 0.0
for i in top100:
    if allans[i] == 1:
        correct_num_100 += 1.0
print(correct_num_100 / 100)

print('P@200:')
top200 = order[:200]
correct_num_200 = 0.0
for i in top200:
    if allans[i] == 1:
        correct_num_200 += 1.0
print(correct_num_200 / 200)

print('P@300:')
top300 = order[:300]
correct_num_300 = 0.0
for i in top300:
    if allans[i] == 1:
        correct_num_300 += 1.0
print(correct_num_300 / 300)

if itchat_run:
tempstr = 'P@100\n' + str(correct_num_100 / 100) + '\n' + 'P@200\n' + str(
    correct_num_200 / 200) + '\n' + 'P@300\n' + str(correct_num_300 / 300)
itchat.send(tempstr, FLAGS.wechat_name)
