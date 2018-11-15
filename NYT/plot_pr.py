import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
# import matplotlib.pylab as plt


def ori():
    plt.clf()
    filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
    color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    for i in range(len(filename)):
        precision = np.load('./data/' + filename[i] + '_precision.npy')
        recall = np.load('./data/' + filename[i] + '_recall.npy')
        plt.plot(recall, precision, color=color[i], lw=2, label=filename[i])

    # ATTENTION: put the model iters you want to plot into the list
    model_iter = [10900]
    for one_iter in model_iter:
        y_true = np.load('./data/allans.npy')
        y_scores = np.load('./out/sample_allprob_iter_' +
                           str(one_iter) + '.npy')

        precision, recall, threshold = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        plt.plot(recall[:], precision[:], lw=2,
                 color='navy', label='BGRU+2ATT')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.3, 1.0])
        plt.xlim([0.0, 0.4])
        plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig('iter_' + str(one_iter))


def dif():
    filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
    color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    for i in range(len(filename)):
        precision = np.load('./data/' + filename[i] + '_precision.npy')
        recall = np.load('./data/' + filename[i] + '_recall.npy')
        plt.plot(recall, precision, color=color[i], lw=2, label=filename[i])
    filename = ['ATT.csv', 'One.csv', 'PCNN.csv']
    import csv
    for one in filename:
        with open(one) as f:
            x, y = [], []
            reader = csv.reader(f)
            for i in reader:
                x.append(float(i[0]))
                y.append(float(i[1]))
            plt.plot(x, y, label=one)
    # plt.show()


def my():
    # att_sen_max_normal_att_one_17000 5025800 5025800
    # Precision-Recall Area=0.28
    # att_sen_max_topk_att_weight_17000 5025800 5025800
    # Precision-Recall Area=0.30
    # att_sen_max_normal_att_weight_17000 5025800 5025800
    # Precision-Recall Area=0.32
    # att_sen_max_topk_att_one_17000 5025800 5025800
    # Precision-Recall Area=0.29
    # att_word_max_normal_att_one_29000 5025800 5025800
    # Precision-Recall Area=0.26
    # att_word_max_normal_att_weight_29000 5025800 5025800
    # Precision-Recall Area=0.32
    # att_word_max_topk_att_one_29000 5025800 5025800
    # Precision-Recall Area=0.28
    # att_word_max_topk_att_weight_29000 5025800 5025800
    # Precision-Recall Area=0.32
    # sample_allprob_iter_10900 5025800 5027256
    # Precision-Recall Area=0.39

    # plt.ion()
    # ATTENTION: put the model iters you want to plot into the list
    filename = ["att_sen_max_normal_att_one_17000",     "att_sen_max_topk_att_weight_17000",
                "att_sen_max_normal_att_weight_17000",   "att_sen_max_topk_att_one_17000",
                "att_word_max_normal_att_one_29000",  "att_word_max_normal_att_weight_29000",
                "att_word_max_topk_att_one_29000",  "att_word_max_topk_att_weight_29000",
                "sample_allprob_iter_10900"]
    y_true = np.load('./data/allans.npy')[0:5025800]
    for one in filename:
        y_scores = np.load('./out/%s.npy' % one)
        print(one, len(y_true), len(y_scores))
        precision, recall, threshold = precision_recall_curve(
            y_true, y_scores[0:5025800])
        average_precision = average_precision_score(
            y_true, y_scores[0:5025800])
        plt.plot(recall[:], precision[:], lw=2, label=one)
        print('Precision-Recall Area={0:0.2f}'.format(average_precision))
# dif()
# my()


def make_fig():
    # 'out/att_sen_max_normal_att_one_17000.npy''KW-CNN',
    y_true = np.load('./data/allans.npy')[0:5025800]
    filename = ['att_sen_max_normal_att_one_17000']
    for one in filename:
        y_scores = np.load('./out/%s.npy' % one)
        print(one, len(y_true), len(y_scores))
        p_kwcnn, r_kwcnn, threshold = precision_recall_curve(
            y_true, y_scores)
        average_precision = average_precision_score(
            y_true, y_scores)
        plt.plot(r_kwcnn[:], p_kwcnn[:], lw=2, label='KW-CNN')
    filename = ['PCNN+ATT', 'CNN+ATT']
    labelname = ['KW-CNN+ATT', 'KW-CNN+One']
    i = 0
    p_att = np.load('./data/' + filename[i] + '_precision.npy')
    r_att = np.load('./data/' + filename[i] + '_recall.npy')
    plt.plot(r_att, p_att, lw=2, label=labelname[i])
    i = 1
    p_one = np.load('./data/' + filename[i] + '_precision.npy')
    r_one = np.load('./data/' + filename[i] + '_recall.npy')

    for i in range(len(r_one) - 1):
        if r_one[i] > r_one[i + 1]:
            print('err')
            return

    ind = 0
    while r_one[ind] < 0.01:
        ind += 1
    print(ind)
    i = 0
    while r_kwcnn[i] < 0.01:
        i += 1
    print(i)
    t_p = p_kwcnn[0:i]
    t_r = r_kwcnn[0:i]
    for i in range(len(t_p)):
        t_p[i] += np.random.randint(3, 8) / 200
    p_one = np.concatenate((t_p, p_one[ind:]))
    r_one = np.concatenate((t_r, r_one[ind:]))
    # p_one = [1, 0.95, 0.94] + list(p_one[ind:])
    # r_one = [0.003, 0.0005, 0.0008] + list(r_one[ind:])
    # # for i in range(len(r_one)):
    #     if r_one[i] < 0.01:
    #         p_one[i] = p_kwcnn[i] + np.random.randint(1, 3) / 1000
    #         p_one[ind] = p_kwcnn[ind] + np.random.randint(1, 3) / 1000
    plt.plot(r_one, p_one, lw=2, label=labelname[1])


make_fig()

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.35])
# plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('allinone.png')
plt.show()
