import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score


def test(settings, wordembedding, itchat_run, FLAGS):
    if itchat_run:
        import itchat

    save_path = settings.model_save_path

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

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

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [m.loss, m.accuracy, m.prob], feed_dict)
                return prob, accuracy

            def printpn(s, order, allans, num):
                print(s)
                top = order[:num]
                correct_num = 0.0
                for i in top:
                    if allans[i] == 1:
                        correct_num += 1.0
                print(correct_num / num)
                return correct_num / num
            # evaluate p@n

            def eval_pn(test_y, test_word, test_pos1, test_pos2, fn):
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(settings.bag_num))):
                    m_st = i * settings.bag_num
                    m_ed = (i + 1) * settings.bag_num
                    prob, accuracy = test_step(test_word[m_st:m_ed],
                                               test_pos1[m_st:m_ed],
                                               test_pos2[m_st:m_ed],
                                               test_y[m_st:m_ed])
                    m_acc = np.mean(np.reshape(
                        np.array(accuracy), (settings.bag_num)))
                    acc.append(m_acc)
                    prob = np.reshape(
                        np.array(prob), (settings.bag_num, settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                eval_y = []
                for i in test_y:
                    eval_y.append(i[1:])
                allans = np.reshape(eval_y, (-1))
                allans = allans[0:len(allprob)]
                order = np.argsort(-allprob)

                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))
                print('saving test result to ', fn)
                np.save(fn, allprob)

                ra = printpn(s='P@100:', order=order, allans=allans, num=100)
                rb = printpn(s='P@200:', order=order, allans=allans, num=200)
                rc = printpn(s='P@300:', order=order, allans=allans, num=300)

                if itchat_run:
                    tempstr = ' P@100 %f \n P@200 %f\n P@200 %f\n' % (
                        ra, rb, rc)
                    itchat.send(tempstr, FLAGS.wechat_name)

            # out of function
            with tf.variable_scope("model"):
                m = network.CNN(is_training=False,
                                word_embeddings=wordembedding,
                                settings=settings)

            saver = tf.train.Saver()

            # 填写训练次数
            testlist = [29000]
            for model_iter in testlist:
                saver.restore(sess, save_path + '-' + str(model_iter))

                print("Evaluating P@N for iter " + str(model_iter))

                if itchat_run:
                    itchat.send("Evaluating P@N for iter %d" %
                                model_iter, FLAGS.wechat_name)

                print('Evaluating P@N for one')
                if itchat_run:
                    itchat.send('Evaluating P@N for one', FLAGS.wechat_name)
                test_y = np.load('./data/pone_test_y.npy')
                test_word = np.load('./data/pone_test_word.npy')
                test_pos1 = np.load('./data/pone_test_pos1.npy')
                test_pos2 = np.load('./data/pone_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2,
                        settings.out_path + 'K_pone_%d.npy' % model_iter)

                print('Evaluating P@N for two')
                if itchat_run:
                    itchat.send('Evaluating P@N for two', FLAGS.wechat_name)
                test_y = np.load('./data/ptwo_test_y.npy')
                test_word = np.load('./data/ptwo_test_word.npy')
                test_pos1 = np.load('./data/ptwo_test_pos1.npy')
                test_pos2 = np.load('./data/ptwo_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2,
                        settings.out_path + 'K_ptwo_%d.npy' % model_iter)

                print('Evaluating P@N for all')
                if itchat_run:
                    itchat.send('Evaluating P@N for all', FLAGS.wechat_name)
                test_y = np.load('./data/pall_test_y.npy')
                test_word = np.load('./data/pall_test_word.npy')
                test_pos1 = np.load('./data/pall_test_pos1.npy')
                test_pos2 = np.load('./data/pall_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2,
                        settings.out_path + 'K_pall_%d.npy' % model_iter)

                print('Evaluating all test data and save data for PR curve')
                if itchat_run:
                    itchat.send(
                        'Evaluating all test data and save data for PR curve', FLAGS.wechat_name)

                test_y = np.load('./data/testall_y.npy')
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2,
                        settings.out_path + 'K_ALL_%d.npy' % model_iter)
