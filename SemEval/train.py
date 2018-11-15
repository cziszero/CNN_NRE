# from constant import *
from data_prepare import *
from constant import *
from text_cnn import TextCNN
import os
import numpy as np
import time
import tensorflow as tf
import datetime

data_dir = 'filter_sizes_4/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

try:
    train_set_x = np.load('train_set_x.npy')
    train_set_y = np.load('train_set_y.npy')
    test_set_x = np.load('test_set_x.npy')
    test_set_y = np.load('test_set_y.npy')
except:
    test_data_path = "TEST_FILE_FULL.TXT"
    train_data_path = "TRAIN_FILE.TXT"

    train_set_size, train_sen_len, train_sens, train_pos, train_rs = read_data(
        train_data_path)
    train_set_x = get_sens_vec(train_sens, train_pos)
    _, train_set_y = get_rels_class(train_rs)
    np.save(data_dir + 'train_set_x', train_set_x)
    np.save(data_dir + 'train_set_y', train_set_y)

    test_set_size, test_sen_len, test_sens, test_pos, test_rs = read_data(
        test_data_path)
    test_set_x = get_sens_vec(test_sens, test_pos)
    _, test_set_y = get_rels_class(test_rs)
    np.save(data_dir + 'test_set_x', test_set_x)
    np.save(data_dir + 'test_set_y', test_set_y)


with tf.Graph().as_default():
    start_time = time.time()
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                      num_filters=FLAGS.num_filters,
                      vec_shape=(FLAGS.sen_len - 2, FLAGS.WF_size *
                                 FLAGS.window_size + 2 * FLAGS.PF_size),
                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                      num_classes=19)
        # 定义训练过程
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # 记录梯度之和稀疏性，便于观察
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 配置models和summaries的存储目录
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "data", data_dir))
        print("Writing to {}\n".format(out_dir))

        # 把loss和accuracy记录下来
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 训练过程summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint 目录.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        train_record = []
        test_record = []

        def train_step(x_text_train, y_batch):
            """
            进行一批训练
            """
            feed_dict = {
                cnn.input_x: x_text_train,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            ops = [train_op, global_step, train_summary_op,
                   cnn.loss, cnn.accuracy, cnn.scores]
            _, step, summaries, loss, accuracy, scores = sess.run(
                ops, feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
            train_record.append((loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            return loss

        def dev_step(x_text_dev, y_batch):
            """
            在测试集上进行评估
            """
            feed_dict = {
                cnn.input_x: x_text_dev,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy, summaries = sess.run(
                [global_step, cnn.loss, cnn.accuracy, dev_summary_op],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            test_record.append((loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)
            return loss

        for X_train, Y_train in get_batches(train_set_x, train_set_y, FLAGS.batch_size, 1001, True):
            loss = accuracy = 0.0
            # X_train, Y_train = np.asarray(X_train), np.asarray(y_train)  #
            # 把X_train和y_train转换成ndarry
            train_loss = train_step(X_train, Y_train)
            current_step = tf.train.global_step(sess, global_step)  # 记步数
            if current_step % FLAGS.evaluate_every == 0:
                print("Evaluation:")
                test_loss = dev_step(test_set_x, test_set_y)
                # 如果测试集loss和训练集loss相差大于early_threshold则退出。
                # if abs(test_loss - train_loss) > FLAGS.early_threshold:
                # exit(0)
            # print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        print("Final Evaluation:")
        test_loss = dev_step(test_set_x, test_set_y)
        pickle.dump(open('./data/test_record.pk', 'wb'), test_record)
        pickle.dump(open('./data/train_record.pk', 'wb'), train_record)
        print("-------------------")
        print("Finished in time %0.3f" % (time.time() - start_time))
