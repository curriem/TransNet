import keras
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from astropy import wcs

def load_data(badpix_dir, model_version):
    model_path = '../model/' + badpix_dir
    y_test = np.load(model_path+'model_y_test_v%i.npy' % model_version)
    x_test = np.load(model_path+'model_x_test_v%i.npy' % model_version)
    #y_test = np.load('../model/sets_oldbadpix/model_y_test_v%i.npy' %
    #                 model_version)
    #x_test = np.load('../model/sets_oldbadpix/model_x_test_v%i.npy' %
    #                 model_version)
    mags_test = np.load(model_path+'model_mag_test_v%i.npy' % model_version)
    #mags_test = np.load('../model/sets_oldbadpix/model_mag_test_v%i.npy' % model_version)
    y_test_reg = []
    for item in y_test:
        if item[0] == 1:
            y_test_reg.append(0)
        elif item[1] == 1:
            y_test_reg.append(1)
        else:
            assert False
    y_test = np.array(y_test_reg)

    return x_test, y_test, mags_test


def true_sn_data(x_test, y_test, mags_test):
    true_sn_args = np.where(y_test == 0)
    y_test = y_test[true_sn_args]
    mags_test = mags_test[true_sn_args]
    x_test = x_test[true_sn_args, :, :, :]
    return x_test, y_test, mags_test, true_sn_args


def load_model(badpix_dir, model_version):
    model_path = '../model/%s/transinet_v%i.h5' % (badpix_dir, model_version)
    model = keras.models.load_model(model_path)
    return model


def evaluate_model(model, x_test):
    y_pred = model.predict_classes(x_test, verbose=1)
    probs = model.predict(x_test)
    scores = probs.T[0]

    return y_pred, probs, scores


def plot_scores_vs_mags(mags, scores, model_version, badpix_dir):
    plt.figure()
    plt.scatter(mags+np.random.normal(size=len(mags))*0.1,
                scores)
    plt.xlabel('mags')
    plt.ylabel('score')
    plt.title('scores vs mags %s' % badpix_dir)
    plt.xlim(24, 28)
    plt.savefig('../plots/%s/model/scores_vs_mags_v%i.pdf' % (badpix_dir,
                                                        model_version))


def plot_mag_roc(mags, scores, y_pred, model_version, badpix_dir):
    plt.figure()
    unique_mags = np.unique(mags)
    threshs = np.arange(0.5, 0.99, 0.1)
    colors = ['blue', 'orange', 'green', 'red', 'grey']
    n = 0
    for mag in unique_mags:
        m = 0
        for thresh in threshs:
            den = float(np.sum(mags == mag))
            num = float(np.sum((mags == mag) & (scores > thresh)))
            if n == 0:
                label = np.around(thresh, 2)
            else:
                label = ''
            plt.scatter(mag, num/den, c=colors[m], label=label, alpha=0.5)
            m += 1
        n += 1
    plt.title(badpix_dir)
    plt.legend(title='thresh')
    plt.xlim(24, 28)
    plt.xlabel('mag')
    plt.ylabel('frac real sne recovered above thresh')
    plt.savefig('../plots/%s/model/mag_roc_v%i.pdf' % (badpix_dir,
                                                 model_version))


def plot_roc(y_test, scores, model_version, badpix_dir):
    num_real_SN_detections = len(np.where(y_test == 0.)[0])
    num_junk = len(y_test) - num_real_SN_detections
    bins = np.arange(0.0, 1.0, 0.01)
    tpr_arr = []
    fpr_arr = []
    for bin in bins:
        tpr_numerator = 0.
        fpr_numerator = 0.
        y_pred_modified = np.ones(len(y_test))
        greater_than_bin = np.greater(scores, bin)
        y_pred_modified[greater_than_bin] = 0
        for n in range(len(y_test)):
            if y_pred_modified[n] == 0. and y_test[n] == 0.:
                tpr_numerator += 1.
            elif y_pred_modified[n] == 0. and y_test[n] == 1.:
                fpr_numerator += 1.
            else:
                continue
        tpr = tpr_numerator / num_real_SN_detections
        fpr = fpr_numerator / num_junk
        tpr_arr.append(tpr)
        fpr_arr.append(fpr)

    plt.figure()
    plt.xlabel('Threshold for SN detection')
    plt.plot(bins, tpr_arr, label='True positive rate')
    plt.plot(bins, fpr_arr, label='False positive rate')
    plt.yscale('log')
    plt.legend()
    plt.title(badpix_dir)
    plt.savefig('../plots/%s/model/roc1_v%i.pdf' % (badpix_dir,
                                              model_version))

    plt.figure()
    plt.plot(fpr_arr, tpr_arr)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.xscale('log')
    plt.title(badpix_dir)
    plt.savefig('../plots/%s/model/roc2_v%i.pdf' % (badpix_dir,
                                              model_version))


def plot_confusion_matrix(y_test, y_pred, model_version, badpix_dir):

    confusion_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(confusion_mat,
               interpolation='nearest',
               cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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
    plt.title(badpix_dir)
    plt.savefig('../plots/%s/model/confusion_matrix_v%i.pdf' % (badpix_dir,
                                                          model_version),
                bbox_inches='tight')


def plot_object_matrix(x_test, y_pred, y_test, probs, model_version,
                       badpix_dir, mags):

    test_inds = np.load('inds_test_rand42_new.npy')
    info = np.load('../data/sets_newbadpix/training_info.npy')
    info = info[test_inds]
    starting_idx = 0
    fig_counter = 0
    mag_sort_args = np.argsort(mags)
    x_test = x_test[mag_sort_args]
    y_pred = y_pred[mag_sort_args]
    y_test = y_test[mag_sort_args]
    probs = probs[mag_sort_args]
    info = info[mag_sort_args]
    print info.shape
    mags = mags[mag_sort_args]
    
    probs_sort_args = np.argsort(probs[:, 1])
    '''
    x_test = x_test[probs_sort_args]
    y_pred = y_pred[probs_sort_args]
    y_test = y_test[probs_sort_args]
    probs = probs[probs_sort_args]
    mags = mags[probs_sort_args]
    '''
    '''
    inds = np.where((mags == 32.) & (probs[:, 0] > 0.97))

    x_test = x_test[inds]
    y_pred = y_pred[inds]
    y_test = y_test[inds]
    probs = probs[inds]
    mags = mags[inds]
    info = info[inds]
    print mags.shape
    print info.shape
    '''
    stop = 0
    for i in range(100):
        if stop:
            break
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        fig.text(0, 0, 'pred(score), true, set, x, y, mag, index')
        obj_counter = 0
        for n in range(10):
            for m in range(10):
                idx = obj_counter + starting_idx

                try:
                    im_set, x, y = info[idx]
                    ax[n, m].imshow(x_test[idx, :, :, 2], cmap='gray')
                    ax[n, m].set_ylabel('%s %s %s' % (im_set, x, y))
                    if y_pred[idx] == 0 and y_test[idx] == 0:
                        color = 'green'
                    elif y_pred[idx] == 1 and y_test[idx] == 0:
                        color = 'red'
                    elif y_pred[idx] == 0 and y_test[idx] == 1:
                        color = 'blue'
                    else:
                        color = 'black'

                    if y_pred[idx] == 0:
                        pred = 'Y'
                    else:
                        pred = 'N'

                    if y_test[idx] == 0:
                        real = 'Y'
                    else:
                        real = 'N'

                    ax[n, m].set_title('%s(%.2f), %i, %i, %i, %.01f' %
                                       (pred, np.max(probs[idx, :]),
                                        int(im_set), int(x), int(y),
                                        mags[idx]),
                                       color=color,
                                       fontsize=7)
                    ax[n, m].axis('off')
                except IndexError:
                    stop = 1
                    break

                obj_counter += 1

            if stop:
                break

        plt.savefig('../plots/%s/model/matrices/result_matrix%i_v%i.pdf'
                    % (badpix_dir, fig_counter, model_version),
                    bbox_inches='tight')
        plt.close()
        fig_counter += 1
        starting_idx += 100


if __name__ == '__main__':
    badpix = sys.argv[1]
    model_version = int(sys.argv[2])
    if badpix == 'new':
        badpix_dir = '/sets_newbadpix/'
    elif badpix == 'old':
        badpix_dir = '/sets_oldbadpix/'
    else:
        assert False
    x_test, y_test, mags_test = load_data(badpix_dir, model_version)

    x_test_sn, y_test_sn, mags_test_sn, real_sn_args = true_sn_data(x_test,
                                                                    y_test,
                                                                    mags_test)
    mags = \
    np.load('/Users/mcurrie/GitRepos/TransiNet/model/%s/model_mag_test_v2.npy'
            % badpix_dir)
    model = load_model(badpix_dir, model_version)

    y_pred, probs, scores = evaluate_model(model, x_test)

    plot_scores_vs_mags(mags_test, scores, model_version, badpix_dir)
    plot_mag_roc(mags_test_sn, scores[real_sn_args],
                 y_pred[real_sn_args], model_version, badpix_dir)

    plot_roc(y_test, scores, model_version, badpix_dir)

    plot_confusion_matrix(y_test, y_pred, model_version, badpix_dir)

    plot_object_matrix(x_test, y_pred, y_test, probs, model_version,
                       badpix_dir, mags)
