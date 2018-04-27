import GPy
import keras
import GPyOpt
import numpy as np
from keras.layers import Flatten, Dropout, BatchNormalization, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


num_classes = 2
num_channels = 2
im_len = 32


class model_setup():
    def __init__(self, first_input=784, last_output=10,
                 l3_conv=512,
                 l3_drop=0.2,
                 dense1=128,
                 conv3_shape=3,
                 validation_split=0.1):
        self.__first_input = first_input
        self.__last_output = last_output
        self.l3_conv = l3_conv
        self.l3_drop = l3_drop
        self.dense1 = dense1
        self.conv3_shape = conv3_shape
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = \
            self.load_data()
        self.__model = self.model()

    def load_data(self):
        data = np.load('../data/training_data.npy')
        labels = np.load('../data/training_labels.npy')

        x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                            random_state=42)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return x_train, x_test, y_train, y_test

    def model(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5),
                         input_shape=(im_len, im_len, num_channels),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.258784285867))
        model.add(Conv2D(128,
                         (4, 4),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.0))
        model.add(Conv2D(self.l3_conv,
                         (self.conv3_shape, self.conv3_shape),
                         activation='relu'))
        model.add(Dropout(self.l3_drop))
        model.add(Flatten())
        model.add(Dense(self.dense1, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        return model

    def model_fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=100,
                         epochs=10,
                         verbose=1,
                         validation_split=self.validation_split,
                         callbacks=[early_stopping])

    def model_evaluate(self):
        self.model_fit()

        evaluation = self.__model.evaluate(self.__x_test,
                                           self.__y_test,
                                           batch_size=100,
                                           verbose=1)
        return evaluation


def run_model(first_input=784, last_output=10,
              l3_conv=512,
              l3_drop=0.2,
              dense1=128,
              conv3_shape=3,
              validation_split=0.1):

    _trans = model_setup(first_input=first_input, last_output=last_output,
                         l3_conv=l3_conv,
                         l3_drop=l3_drop,
                         dense1=dense1,
                         conv3_shape=conv3_shape,
                         validation_split=validation_split)

    eval = _trans.model_evaluate()
    return eval


bounds = [{'name': 'validation_split', 'type': 'continuous',  'domain': (0.0,
                                                                         0.3)},
          {'name': 'l3_drop',          'type': 'continuous',  'domain': (0.0,
                                                                         0.3)},
          {'name': 'l3_conv',           'type': 'discrete',    'domain': (32,
                                                                          64,
                                                                          128,
                                                                          256,
                                                                          512,
                                                                          1024
                                                                          )},
          {'name': 'dense1',           'type': 'discrete',    'domain': (32,
                                                                         64,
                                                                         128,
                                                                         256,
                                                                         512,
                                                                         1024
                                                                         )},
          {'name': 'conv3_shape',      'type': 'discrete',    'domain': (2,
                                                                         3,
                                                                         4,
                                                                         5)}]


def f(x):
    print x
    evaluation = run_model(l3_drop=float(x[:, 1]),
                           l3_conv=int(x[:, 2]),
                           dense1=int(x[:, 3]),
                           conv3_shape=int(x[:, 4]),
                           validation_split=float(x[:, 0]))

    print "LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1])
    print evaluation
    return evaluation[0]


# optimizer
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

opt_model.run_optimization(max_iter=10)

print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
""".format(bounds[0]["name"], opt_model.x_opt[0],
           bounds[1]["name"], opt_model.x_opt[1],
           bounds[2]["name"], opt_model.x_opt[2],
           bounds[3]["name"], opt_model.x_opt[3],
           bounds[4]["name"], opt_model.x_opt[4]))

print "optimized loss: {0}".format(opt_model.fx_opt)

opt_model.x_opt


'''
optimized parameters for cnn v0:
validation_split        0.0939408047765
l1_drop                 0.258784285867
l1_conv                 32.0
dense1                  64.0
conv1_shape             5.0
pool1_shape             2.0
batch_size              100.0
epochs                  10.0


optimized params for cnn v1:
validation_split        0.0
l2_drop                 0.0
l2_conv                 128.0
dense1                  256.0
conv2_shape             4.0

optimized params for cnn v2:
validation_split        0.0
l3_drop                 0.3
l3_conv                 256.0
dense1                  128.0
conv2_shape             3.0
'''
