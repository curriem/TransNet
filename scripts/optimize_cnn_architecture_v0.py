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
                 l1_conv=512,
                 l2_conv=512,
                 l1_drop=0.2,
                 l2_drop=0.2,
                 dense1=128,
                 dense2=32,
                 conv1_shape=3,
                 pool1_shape=2,
                 batch_size=100,
                 epochs=10,
                 validation_split=0.1):
        self.__first_input = first_input
        self.__last_output = last_output
        self.l1_conv = l1_conv
        self.l1_drop = l1_drop
        self.dense1 = dense1
        self.conv1_shape = conv1_shape
        self.pool1_shape = pool1_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = \
            self.load_data()
        self.__model = self.baseline_model()

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

    def baseline_model(self):
        model = Sequential()
        model.add(Conv2D(self.l1_conv, (self.conv1_shape, self.conv1_shape),
                         input_shape=(im_len, im_len, num_channels),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.pool1_shape, self.pool1_shape)))
        model.add(Dropout(self.l1_drop))
        model.add(Flatten())
        model.add(Dense(self.dense1, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        return model

    def model_fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=1,
                         validation_split=self.validation_split,
                         callbacks=[early_stopping])

    def model_evaluate(self):
        self.model_fit()

        evaluation = self.__model.evaluate(self.__x_test,
                                           self.__y_test,
                                           batch_size=self.batch_size,
                                           verbose=1)
        return evaluation


def run_model(first_input=784, last_output=10,
              l1_conv=512,
              l1_drop=0.2,
              dense1=128,
              conv1_shape=3,
              pool1_shape=2,
              batch_size=100, epochs=10,
              validation_split=0.1):

    _trans = model_setup(first_input=first_input, last_output=last_output,
                         l1_conv=l1_conv,
                         l1_drop=l1_drop,
                         dense1=dense1,
                         conv1_shape=conv1_shape,
                         pool1_shape=pool1_shape,
                         batch_size=batch_size, epochs=epochs,
                         validation_split=validation_split)

    eval = _trans.model_evaluate()
    return eval


bounds = [{'name': 'validation_split', 'type': 'continuous',  'domain': (0.0,
                                                                         0.3)},
          {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0,
                                                                         0.3)},
          {'name': 'l1_conv',           'type': 'discrete',    'domain': (32,
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
          {'name': 'conv1_shape',      'type': 'discrete',    'domain': (2,
                                                                         3,
                                                                         4,
                                                                         5)},
          {'name': 'pool1_shape',      'type': 'discrete',    'domain': (2,
                                                                         3)},
          {'name': 'batch_size',       'type': 'discrete',    'domain': (10,
                                                                         100,
                                                                         500,
                                                                         1000
                                                                         )},
          {'name': 'epochs',           'type': 'discrete',    'domain': (5, 10,
                                                                         20)}]


def f(x):
    print x
    evaluation = run_model(l1_drop=float(x[:, 1]),
                           l1_conv=int(x[:, 2]),
                           dense1=int(x[:, 3]),
                           conv1_shape=int(x[:, 4]),
                           pool1_shape=int(x[:, 5]),
                           batch_size=int(x[:, 6]),
                           epochs=int(x[:, 7]),
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
\t{10}:\t{11}
\t{12}:\t{13}
\t{14}:\t{15}
""".format(bounds[0]["name"], opt_model.x_opt[0],
           bounds[1]["name"], opt_model.x_opt[1],
           bounds[2]["name"], opt_model.x_opt[2],
           bounds[3]["name"], opt_model.x_opt[3],
           bounds[4]["name"], opt_model.x_opt[4],
           bounds[5]["name"], opt_model.x_opt[5],
           bounds[6]["name"], opt_model.x_opt[6],
           bounds[7]["name"], opt_model.x_opt[7]))

print "optimized loss: {0}".format(opt_model.fx_opt)

opt_model.x_opt


'''
optimized parameters:
validation_split        0.0939408047765
l1_drop                 0.258784285867
l1_conv                 32.0
dense1                  64.0
conv1_shape             5.0
pool1_shape             2.0
batch_size              100.0
epochs                  10.0
'''


