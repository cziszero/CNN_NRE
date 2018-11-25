import numpy as np


class Dataset(object):
    """
    对原始数据进行处理, 并支持生成一批训练数据等操作
    """

    def __init__(self, data_path='data/train_file.txt'):
        self.size = 0
        self.rs = []
        self.sens = []
        self.pos = []
        self.r2n = dict()
        self.n2r = dict()

        self.read_rel_class()
        self.read_raw_data(data_path=data_path)

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
        rs, pos, sens = self.rs, self.pos, self.sens

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
                    word = tt[j][:-1] if tt[j][-1] in split_char else tt[j]
                    word = word[1:] if len(word) > 0 and word[0] in split_char else word

                if word != '':
                    s_words.append(word)
                    ind += 1
                j += 1

            sens.append(s_words)
            sen_maxlen = max(sen_maxlen, ind)
            pos.append((e1, e2))

        self.size = len(sens)

        exampleID = np.random.randint(0, len(ss))
        print("example %d\n %s \n %s" % (exampleID, ss[exampleID], self.n2r[rs[exampleID]]))
        s_words = sens[exampleID]
        p = pos[exampleID]
        print(s_words, '\n', p)
        print(s_words[p[0]], ' ', s_words[p[1]])

        print('Total %d samples, Max Length %d, ' % (self.size, sen_maxlen))


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
        import pickle as pk

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
        print("There are %d words in both train and test dataset"%(len(words)))
        print("%d words is't in GoogleNews-vectors"%(not_in_dict))
