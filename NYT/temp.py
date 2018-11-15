from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Input, Conv1D
import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.models import Model


fixlen = 70
pos_size = 5
word_size = 50
bag_num = 50
filter_height = 3
n1 = 200

t_i = Input(shape=(5,))
t_o = Dense(units=1)(t_i)
model = Model(inputs=[t_i], outputs=t_o)
model.compile(loss='mse',
              optimizer='sgd')
x = np.random.randint(1, 100, (10, 5))
y = np.sum(x, 1)
model.fit(x, y)

sens = Input(shape=(fixlen, pos_size * 2 + word_size))

pos = []
for i in range(0, 10):
    a = np.random.randint(70)
    b = np.random.randint(70)
    pos.append((min(a, b), max(a, b)))

ss = []
for i in range(len(pos)):
    p = pos[i]
    s1 = K.max(sens[i, 0:p[0], :], 0)
    s2 = K.max(sens[i, p[0]:p[1], :], 0)
    s3 = K.max(sens[i, p[1]:70, :], 0)
    s = K.concatenate([s1, s2, s3])
    s = K.reshape(s, [1, -1])
    ss.append(s)
ss = K.concatenate(ss, 0)
la = Lambda(lambda x: x)(ss)
d = Dense(3)(la)


bag_ind = Input(shape=(bag_num,), dtype='int32')

conv = Conv1D(filters=n1, kernel_size=(filter_height), padding="same")(sens)
