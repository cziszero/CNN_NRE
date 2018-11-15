from keras.layers import Input
from keras.models import Model
from keras.engine import Layer
from keras import backend as K
import numpy as np


class MDense(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.name = 'WTF'
        super(MDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        print("input_shape %s, input_dim %s" %
              (str(input_shape), str(input_dim)))
        self.kernel = self.add_weight(
            (input_dim, self.units), initializer='uniform')
        self.bias = self.add_weight((self.units,), initializer='uniform')
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        output = K.bias_add(output, self.bias)
        # K.eval(inputs)
        print(inputs.shape[0])
        # print(i)
        print('inputs ', inputs)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        print("input_shape %s, output_shape %s" %
              (str(input_shape), str(output_shape)))
        return tuple(output_shape)

t_i = Input(shape=(5,))
t_o = MDense(units=1)(t_i)
model = Model(inputs=[t_i], outputs=t_o)
model.compile(loss='mse',
              optimizer='sgd')
x = np.random.randint(1, 100, (10, 5))
y = np.sum(x, 1)
model.fit(x, y, epochs=3, batch_size=1)
