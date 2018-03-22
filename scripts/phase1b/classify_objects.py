import sys
import keras
import numpy as np
import matplotlib.pyplot as plt



def plot_candidate(candidate, score, info):
    print info
    plt.figure()
    stack = np.hstack((candidate[:, :, 0], candidate[: , :, 1]))
    plt.imshow(stack, cmap='gray')
    plt.title('%s %s %s' % (score, str(info[0]), str(info[1])))
    plt.axis('off')
def main():
    model_path = '../../model/transinet_v1.h5'
    model = keras.models.load_model(model_path)

    set_num = sys.argv[1]
    data_path = '/Users/mcurrie/GitRepos/TransiNet/data/'
    cand_path = data_path + 'object_candidates_set_%s.npy' % set_num
    info_path = data_path + 'candidate_info_set_%s.npy' % set_num
    info = np.load(info_path)
    data = np.load(cand_path)
    class_pred = model.predict_classes(data, verbose=1)
    probs = model.predict(data)
    prob_of_SN = probs.T[0]

    candidate_inds = np.where(prob_of_SN > 0.97)
    print 'Num candidates:', len(candidate_inds[0])
    for ind in candidate_inds[0]:
        plot_candidate(data[ind, :, :, :], prob_of_SN[ind], info[ind])
    plt.show()

main()
