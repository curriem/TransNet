from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import commands
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_play(features, labels, mode):
    """ model for basic CNN"""

    # input layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # convolutional layer 1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)

    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    # convolutional layer 2
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)

    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(inputs=dropout,
                             units=3)

    predictions = {
        # generate predictions for PREDICT and EVAL mode
        "classes": tf.argmax(input=logits, axis=1),

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    # for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # for EVAL mode
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main():

    # delete junk in play_model dir
    commands.getoutput('rm ../play_model/*')

    # load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    data = np.load('../data/gauss_data.npy')
    labels = np.load('../data/gauss_labels.npy')
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        random_state=42)

    # create estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_play,
                                              model_dir='../play_model')
    # set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                        y=y_train,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,
                           steps=10000,
                           hooks=[logging_hook])

    # evaluate the model and print the results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},
                                                       y=y_test,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


main()
