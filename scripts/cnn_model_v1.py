import keras
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import roc_curve, auc, recall_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import logging
import sep 
import pyfits
import cPickle as pickle


def baseline_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def deeper_model(input_shape):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def mnist_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
.(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def vgg_like_model(input_shape):
    model = Sequential()
    model.add(Conv2D(96, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))




def plot_confusion_matrix(
                          data, title='_Confusion Matrix_', cmap=plt.cm.Blues, name=''):
    logging.debug("Producing Chart - Confusion Matrix - {}".format(name))
    plt.imshow(data,
               interpolation='nearest',
               cmap=cmap)
    plt.title(title)
    plt.colorbar()
    labels = np.array(['Negative',
                       'Positive'])
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,
                          labels,
                          rotation=45)
    plt.yticks(tick_marks,
                          labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("../plots/" + name + title + '.png',
                bbox_inches='tight')


def plot_roc_curve(fpr, tpr, roc_auc, name=''):
    logging.debug("Producing Chart - ROC Curve - {}".format(name))
    plt.figure()
    plt.plot(fpr,
             tpr,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             'k--')
    plt.xlim([0.0,
                       1.0])
    plt.ylim([0.0,
                       1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("../plots/" + name + '_ROC_Curve.png',
                         bbox_inches='tight')



def plot_roc_curve2(prob_of_SN, real_labels):

    plt.figure()
    num_real_SN_detections = len(np.where(real_labels == 0.)[0])
    num_junk = len(real_labels) - num_real_SN_detections
    bins = np.arange(0.01, 1.0, 0.01)
    tpr_arr = []
    fpr_arr = []
    for bin in bins:
        tpr_numerator = 0.
        fpr_numerator = 0.
        pred_labels = np.ones(len(real_labels))
        greater_than_bin = np.greater(prob_of_SN, bin)
        pred_labels[greater_than_bin] = 0
        for n in range(len(real_labels)):
            if pred_labels[n] == 0. and real_labels[n] == 0.:
                tpr_numerator += 1.
            elif pred_labels[n] == 0. and real_labels[n] == 1.:
                fpr_numerator += 1.
            else:
                continue

        #fpr, tpr, _ = roc_curve(real_labels, pred_labels)
        tpr = tpr_numerator / num_real_SN_detections
        fpr = fpr_numerator / num_junk
        print 'RATES:', tpr, fpr
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)

    plt.xlabel('Threshold for SN detection')
    plt.ylim(-0.1, 1.1)
    plt.plot(bins, tpr_arr, label='True positive rate')
    plt.plot(bins, fpr_arr, label='False positive rate')
    plt.legend()
    plt.savefig('../plots/new_ROC.pdf')
    
    plt.figure()
    plt.plot(fpr_arr, tpr_arr)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.xscale('log')
    plt.savefig('../plots/new_ROC2.pdf')

batch_size = 100
num_classes = 2
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

num_channels = 2

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
data = np.load('../data/training_data.npy')
labels = np.load('../data/training_labels.npy')
mags = np.load('../data/training_mags.npy')
print data.shape



print labels.shape
x_train, x_test, y_train, y_test, mags_train, mags_test = train_test_split(data,
                                                    labels,
                                                                           mags,
                                                    random_state=42)

#x_train = x_train[:,:,:,0]
#x_test = x_test[:,:,:,0]

if int(sys.argv[1]):
    for n in np.random.randint(len(y_train), size=20):
        fig, axes = plt.subplots(1, 1)
        cb = axes.imshow(x_train[n, :, :, 0], cmap='gray')
        #axes[1].imshow(x_train[n, :, :, 1], cmap='gray')
        fig.colorbar(cb)
        plt.title(y_train[n])

    plt.show()
    assert False


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], num_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], num_channels, img_rows, img_cols)
    # artifacts = artifacts.reshape(artifacts.shape[0], num_channels, img_rows,
    #                             img_cols)
    input_shape = (num_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,
                              num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
    #artifacts = artifacts.reshape(artifacts.shape[0], img_rows, img_cols,
    #                              num_channels)
    input_shape = (img_rows, img_cols, num_channels)
#input_shape = (img_rows, img_cols, num_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print 'x_train shape:',  x_train.shape
print x_train.shape[0],  'train samples'
print x_test.shape[0],  'test samples'


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#model = baseline_model(input_shape)
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5,
#          batch_size=100, verbose=1)
#scores = model.evaluate(x_test, y_test, verbose=1)
#print("CNN Error: %.2f%%" % (100-scores[1]*100))


model = deeper_model(input_shape)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5,
                      batch_size=100, verbose=1)
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

#model = mnist_model(input_shape)
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,
#                      batch_size=100, verbose=1)
#scores = model.evaluate(x_test, y_test, verbose=1)
#print("CNN Error: %.2f%%" % (100-scores[1]*100))

'''

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
#model.add(Conv2D(128, (2, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
fit = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
'''
score = model.evaluate(x_test, y_test, verbose=0)
print score
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
model.save('../model/transinet_v2.h5')
np.save('model_x_test.npy', x_test)
np.save('model_y_test.npy', y_test)
np.save('model_mag_test.npy', mags_test)

y_pred = model.predict_classes(x_test, verbose=1)
y_test_reg = []
for item in y_test:
    if item[0] == 1:
        y_test_reg.append(0)
    elif item[1] == 1:
        y_test_reg.append(1)
    #elif item[2] == 1:
    #    y_test_reg.append(2)
    else:
        assert False

y_test = np.array(y_test_reg)
'''
y_pred_onehot = []
for item in y_pred:
    if item == 0:
        y_pred_onehot.append(np.array([1., 0.]))
    elif item == 1:
        y_pred_onehot.append(np.array([0., 1.]))
    else:
        assert False

'''
print y_pred
print y_test



print y_test.shape, y_pred.shape
acc_score = accuracy_score(y_test, y_pred)
print 'HERE',acc_score

F1_score = f1_score(y_test, y_pred)
print F1_score
precision = precision_score(y_test, y_pred)
print precision

logLoss = log_loss(y_test, y_pred)

recall = recall_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

roc_auc = auc(fpr, tpr)

plot_confusion_matrix(cm, name='SNe')
plot_roc_curve(fpr, tpr, roc_auc, name='SNe')
'''
plt.figure()
plt.plot(fit.history['val_acc'])
plt.ylabel('val_acc')
plt.xlabel('epoch')
plt.title('Simulated SNe validation accuracy')
plt.savefig('../plots/sim_val_acc.pdf')
plt.show()
'''



pred_cl = model.predict_classes(x_test, verbose=1)
print pred_cl
'''
for n in range(20):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_test[n, :, :, 0], cmap='gray')
    axes[1].imshow(x_test[n, :, :, 1], cmap='gray')
    plt.title(y_test[n])

plt.show()
'''
'''
probs = model.predict(x_test)
prob_of_SN = probs.T[0]
print prob_of_SN.shape
print prob_of_SN
plot_roc_curve2(prob_of_SN, y_test)
starting_idx = 0
fig_counter = 0


print probs
p = probs.T[0, :]
data = {}
data['images'] = x_test
data['predicted_class'] = pred_cl
data['score'] = p
data['true_class'] = y_test
pickle.dump(data, open('./run_data.p', 'wb'))
assert False
print p
print pred_cl
print len(p) 
print len(pred_cl)
for n in range(len(p)):
    if p[n] > 0.97 and y_test[n] == 1:
        plt.figure()
        stack = np.hstack((x_test[n, :, :, 0], x_test[n, :, :, 1]))
        plt.imshow(stack, cmap='gray')
        plt.axis('off')
        plt.title('%.4f'%p[n])
        plt.savefig('../plots/candidate%i.pdf'%n)
        plt.close()
assert False
'''
c = 0
'''
fig, ax = plt.subplots(10,10, figsize=(20,20))
for arg in above97_args:
    ax[n,m].imshow(x_test[arg, :, :, 0], cmap='gray')
    if pred_cl[arg] == 0 and y_test[arg] == 0:
        color='green'
    elif pred_cl[arg] == 1 and y_test[arg] == 0:
        color='red'
    elif pred_cl[arg] == 0 and y_test[arg] == 1:
        color='blue'
        plt.figure()
        plt.imshow(x_test[arg, :, :, 0], cmap='gray')
        plt.savefig('../plots/false_pos%i' % c)
        plt.close()
        c+=1
    else:
        color='black'
    if pred_cl[arg] == 0:
        pred = 'Y'
    else:
        pred = 'N'

    if y_test[arg] == 0:
        real = 'Y'
    else:
        real = 'N'
    ax[n,m].set_title('pred: %s (%.3f), real: %s' % (pred,
                                                     np.max(probs[arg,
                                                                  :]),
                                                     real),
                     color=color,
                     fontsize=9)
    ax[n,m].axis('off')

plt.savefig('../plots/result_matrix_above97.pdf', bbox_inches='tight')
'''
probs = model.predict(x_test)
starting_idx = 0
fig_counter = 0
for i in range(30):

    fig, ax = plt.subplots(10,10, figsize=(20,20))
    counter = 0
    for n in range(10):
        for m in range(10):
            idx = counter + starting_idx
            try:

                ax[n,m].imshow(x_test[idx, :, :, 0], cmap='gray')
                if pred_cl[idx] == 0 and y_test[idx] == 0:
                    color='green'
                elif pred_cl[idx] == 1 and y_test[idx] == 0:
                    color='red'
                elif pred_cl[idx] == 0 and y_test[idx] == 1:
                    color='blue'
                else:
                    color='black'

                
                if pred_cl[idx] == 0:
                    pred = 'Y'
                else:
                    pred = 'N'

                if y_test[idx] == 0:
                    real = 'Y'
                else:
                    real = 'N'
                ax[n,m].set_title('pred:%s (%.2f), real:%s %i' % (pred,
                                            np.max(probs[idx,:]),
                                            real, idx),
                                 color=color,
                                 fontsize=9)
                ax[n,m].axis('off')
            except:
                pass
            counter +=1 
    plt.savefig('../plots/result_matrix%i.pdf' % fig_counter, bbox_inches='tight')
    fig_counter += 1
    starting_idx += 100



