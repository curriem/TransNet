import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def cnn_v0(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_v1(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Conv2D(128, (4, 4),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_v2(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Conv2D(128, (4, 4),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.0))
    model.add(Conv2D(256, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


# v3 has not been optimized yet!!!!!
def cnn_v3(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Conv2D(128, (4, 4),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.0))
    model.add(Conv2D(256, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (2, 2),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# v4 has not been optimized yet!!!!!
def cnn_v4(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Conv2D(128, (4, 4),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (2, 2),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# v5 has not been optimized yet!!!!!
def cnn_v5(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     input_shape=input_shape,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.26))
    model.add(Conv2D(128, (4, 4),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (3, 3),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (2, 2),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(1028, (2, 2),
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


badpix = sys.argv[1]
if badpix == 'new':
    badpix_dir = 'sets_newbadpix'
elif badpix == 'old':
     badpix_dir = 'sets_oldbadpix'
else:
    assert False

batch_size = 100
num_classes = 2
epochs = 10
model_version = int(sys.argv[2])
models = [cnn_v0, cnn_v1, cnn_v2, cnn_v3, cnn_v4, cnn_v5]
img_rows, img_cols = 32, 32
num_channels = 3
input_shape = (img_rows, img_cols, num_channels)

data = np.load('../data/%s/training_data.npy' % badpix_dir)
labels = np.load('../data/%s/training_labels.npy' % badpix_dir)
mags = np.load('../data/%s/training_mags.npy' % badpix_dir)

inds = range(data.shape[0])

x_train, x_test,\
    y_train, y_test,\
    mags_train, mags_test, \
    ind_train, ind_test = train_test_split(data,
                                             labels,
                                             mags,
                                           inds,
                                             random_state=42)

np.save('inds_train_rand42_%s.npy' % badpix, ind_train)
np.save('inds_test_rand42_%s.npy' % badpix, ind_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print x_train.shape[0],  'train samples'
print x_test.shape[0],  'test samples'

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = models[model_version](input_shape)
early_stopping = EarlyStopping(patience=1, verbose=1)
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=epochs,
          batch_size=batch_size, verbose=1,
          callbacks=[early_stopping])
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save('../model/%s/transinet_v%i.h5' % (badpix_dir, model_version))
np.save('../model/%s/model_x_test_v%i.npy' % (badpix_dir, model_version),
        x_test)
np.save('../model/%s/model_y_test_v%i.npy' % (badpix_dir, model_version),
        y_test)
np.save('../model/%s/model_mag_test_v%i.npy' % (badpix_dir, model_version),
        mags_test)
