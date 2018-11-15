import pickle
import numpy as np


def read_data(data_path='data/TRAIN_FILE.TXT', rs_filepath='data/rel_to_class.pk'):
    """
    return ss,rs,sens, pos
    """
    # split_char = set()
    # for i in ss:
    #     for j in i:
    #         if not j.isalpha():
    #             split_char.add(j)

    split_char = {"'", '(', '\n', ')', ':', '*', '?',  '+', '=', '~',
                  ';', '%', '"', '&', '.', '-', '\t', ',', '_', ' ', '!', '$', '#'}
    f = open(data_path, 'r')
    r2n = pickle.load(open(rs_filepath, 'rb'))
    ss = []
    rs = []
    while True:
        s = f.readline()
        if s == '':
            break
        s = s[s.find('\t'):-2].lower()
        for j in split_char:
            s = s.replace(j, ' ')
        r = f.readline()[:-1]
        r = r2n[r]
        _ = f.readline()
        _ = f.readline()
        ss.append(s)
        rs.append(r)

    exampleID = np.random.randint(0, len(ss))
    print("example %d\n" % (exampleID), ss[exampleID], '\n', rs[exampleID])

    sen_len = 0
    pos = []
    sens = []
    for sen in ss:
        tt = sen.split()
        s_words = []
        ind = -1
        j = -1
        while j < len(tt) - 1:
            ind += 1
            j += 1
            if tt[j].find('<e1>') > -1:
                e1 = ind
                word = tt[j].replace('<e1>', '')
                while tt[j].find('</e1>') == -1:
                    j += 1
                    word += "_" + tt[j]
                word = word.replace('</e1>', '')
                s_words.append(word)
            elif tt[j].find('<e2>') > -1:
                e2 = ind
                word = tt[j].replace('<e2>', '')
                while tt[j].find('</e2>') == -1:
                    j += 1
                    word += "_" + tt[j]
                word = word.replace('</e2>', '')
                s_words.append(word)
            else:
                s_words.append(tt[j])
        sens.append(s_words)
        sen_len = max(sen_len, ind)
        pos.append((e1, e2))
    set_size = len(ss)

    s1 = 0
    s2 = 0
    s3 = 0
    for i in range(0, set_size):
        s = pos[i]
        s1 = max(s1, s[0] + 1)
        s2 = max(s2, s[1] - s[0])
        s3 = max(s3, len(sens[i]) - 1 - s[1])

    s_words = sens[exampleID]
    p = pos[exampleID]
    print(s_words, '\n', p)
    print(s_words[p[0]], ' ', s_words[p[1]])

    return set_size, sen_len, (s1, s2, s3), sens, pos, rs


def w2v_pre_data():
    data_path = "all.txt"
    senfilename = 'train_word2vec.txt'
    entityfilename = 'entitys.txt'
    mode = 'w'
    set_size, sen_len, seg, sens, pos, rs = read_data(data_path=data_path)
    fa = open(senfilename, mode, encoding='utf-8')
    fb = open(entityfilename, mode, encoding='utf-8')
    for sen, p, r in zip(sens, pos, rs):
        for word in sen:
            fa.write(word + ' ')
        fa.write('\n')
        fb.write(sen[p[0]] + ',' + sen[p[1]] + ',%d\n' % (r))
    fb.close()
    fa.close()
