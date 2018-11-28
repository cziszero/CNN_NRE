# coding=utf-8

import numpy as np
import pickle as pk
import os


class Dataset(object):
    """
    对原始数据进行处理, 并支持生成一批训练数据等操作
    """

    def __init__(self, data_path, batch_size=40, isRandom=False):
        np.random.seed(0)

        self.MAX_SEN_LEN = 86  # 最大句子长度
        self.batch_size = batch_size
        self.size = 0
        self.idx = 0
        self.rels = []
        self.sens = []
        self.sens_vec = []
        self.pos = []
        self.r2n = dict()
        self.n2r = dict()

        self.read_rel_class()
        self.read_raw_data(data_path=data_path)
        if isRandom:
            self.word_embeddings = pk.load(open('data/word_embeddings_random_dict.pk', 'rb'))
        else:
            self.word_embeddings = pk.load(open('data/word_embeddings_dict.pk', 'rb'))
        self.lPF, self.rPF = np.random.rand(86 * 2, 5), np.random.rand(86 * 2, 5)
        self.mk_sens_vec()
        self._shuffle()

    def read_rel_class(self, filename='data/rel_to_class.txt'):
        """
        读取关系和关系id的关系
        :param filename: 存储关系和关系id的文件名
        :return:
        """
        r2n = self.r2n
        with open(filename, 'r', encoding='utf-8') as f:
            while True:
                l = f.readline().strip()
                if l == '': break;
                l = l.split(':')
                r2n[l[0]] = int(l[1])
                self.n2r[int(l[1])] = l[0]

    def read_raw_data(self, data_path):
        """
        对原始文件进行处理, 得到按单词分好的句子, 和对应的关系, 即实体的位置
        :param data_path: 原始数据文件
        :return:
        """
        # split_char = set()
        # for i in ss:
        #     for j in i:
        #         if not j.isalpha():
        #             split_char.add(j)

        split_char = {"'", '(', '\n', ')', ':', '*', '?', '+', '=', '~',
                      ';', '%', '"', '&', '.', '-', '\t', ',', '_', ' ', '!', '$', '#'}

        r2n = self.r2n
        ss = []
        sen_maxlen = 0
        rs, pos, sens = [], [], self.sens

        with open(data_path, 'r', encoding='utf-8') as f:
            while True:
                s = f.readline()
                if s == '': break
                s = s[s.find('\t') + 2:-3].lower()
                # for j in split_char:
                #     s = s.replace(j, ' ')
                r = f.readline()[:-1]
                r = r2n[r]
                f.readline()
                f.readline()
                ss.append(s)
                rs.append(r)

        for sen in ss:
            tt = sen.split()
            s_words = []
            ind, j = 0, 0
            while j < len(tt):
                if tt[j].find('<e1>') > -1:
                    e1 = ind
                    word = tt[j].replace('<e1>', '')
                    while tt[j].find('</e1>') == -1:
                        j += 1
                        word += "_" + tt[j]
                    word = word.replace('</e1>', '')
                elif tt[j].find('<e2>') > -1:
                    e2 = ind
                    word = tt[j].replace('<e2>', '')
                    while tt[j].find('</e2>') == -1:
                        j += 1
                        word += "_" + tt[j]
                    word = word.replace('</e2>', '')
                else:
                    word = tt[j]

                while len(word) > 0 and word[-1] in split_char: word = word[:-1]
                while len(word) > 0 and word[0] in split_char: word = word[1:]
                if word != '':
                    s_words.append(word)
                    ind += 1
                j += 1

            sens.append(s_words)
            sen_maxlen = max(sen_maxlen, ind)
            pos.append((e1, e2))

        sen_maxlen += 1 #前面要添加一个padding, 以保证每一段都有内容
        self.size = len(sens)
        self.pos = np.asarray(pos) + 1
        self.rels = np.asarray(rs)

        exampleID = np.random.randint(0, len(ss))
        print("example %d\n %s \n %s" % (exampleID, ss[exampleID], self.n2r[rs[exampleID]]))
        s_words = sens[exampleID]
        p = pos[exampleID]
        print(s_words, '\n', p)
        print(s_words[p[0]], ' ', s_words[p[1]])

        print('Total %d samples, Max Length %d, ' % (self.size, sen_maxlen))

    def mk_sens_vec(self):
        """
        将字符串形式的数据站换为词向量形式的数据
        :return:
        """
        F_PAD = np.zeros(50 + 5 * 2, dtype=np.float32)
        sens_vec = []
        for i, sen in enumerate(self.sens):
            _sen_vec = []
            _sen_vec.append(F_PAD)
            for j, word in enumerate(sen):
                wf = self.word_embeddings[word]
                lpf = self.lPF[self.pos[i][0] - j + self.MAX_SEN_LEN]
                rpf = self.rPF[self.pos[i][0] - j + self.MAX_SEN_LEN]
                _sen_vec.append(np.concatenate([wf, lpf, rpf]))
            while len(_sen_vec) < self.MAX_SEN_LEN:  # 填充至最大句子长度
                _sen_vec.append(F_PAD)
            sens_vec.append(_sen_vec)
        self.sens_vec = np.asarray(sens_vec)

    def __iter__(self):
        return self

    def _shuffle(self):
        """
        将数据进行混洗
        :return: None
        """
        _idx = np.random.permutation(self.size)
        self.rels = self.rels[_idx]
        self.sens_vec = self.sens_vec[_idx]
        self.pos = self.pos[_idx]

    def __next__(self):
        """
        生成一个batch迭代的数据
        :return:
        """
        if self.idx + self.batch_size >= self.size:
            self.idx = 0
            self._shuffle()
            raise StopIteration
        _idx = slice(self.idx, self.idx + self.batch_size)
        x = self.sens_vec[_idx]
        y = self.rels[_idx]
        pos = self.pos[_idx]
        self.idx += self.batch_size
        return x, y, pos


if __name__ == '__main__':
    """
    进行简单测试, 并生成word_embeddings字典
    """
    print('=' * 10, 'Loading Train Dataset', '=' * 10)
    train_data = Dataset(data_path='data/train_file.txt')
    for i, (e1, e2) in enumerate(train_data.pos):
        if e1 > e2:
            print(train_data.sens[i])

    print('=' * 10, 'Loading Test Dataset', '=' * 10)
    test_data = Dataset(data_path='data/test_file.txt')
    for i, (e1, e2) in enumerate(test_data.pos):
        if e1 > e2:
            print(test_data.sens[i])

    building_word_embeddings = False
    if building_word_embeddings:
        import gensim

        words = set()
        word_embeddings = {}

        for sen in train_data.sens + test_data.sens:
            for w in sen:
                words.add(w)
        words = list(words)
        words.sort()

        not_in_dict = 0
        w2v_g = gensim.models.KeyedVectors.load_word2vec_format('d:/GoogleNews-vectors-negative300.bin', binary=True)
        for w in words:
            try:
                word_embeddings[w] = w2v_g[w][0:50]
            except KeyError:
                word_embeddings[w] = np.random.rand(50)
                not_in_dict += 1

        pk.dump(word_embeddings, open('data/word_embeddings_dict.pk', 'wb'))
        with open('data/word_embeddings_dict.txt', 'w', encoding='utf-8') as f:
            for w in words:
                f.write(w + ' : ' + str(list(word_embeddings[w])) + '\n')
        print("There are %d words in both train and test dataset" % (len(words)))
        print("%d words is't in GoogleNews-vectors" % (not_in_dict))

        word_embeddings = {}
        for w in words:
            word_embeddings[w] = np.random.rand(50)
        pk.dump(word_embeddings, open('data/word_embeddings_random_dict.pk', 'wb'))
