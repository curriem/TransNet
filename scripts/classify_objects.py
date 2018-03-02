import sys
import keras
import numpy as np
import matplotlib.pyplot as plt



def plot_candidate(candidate, score):
    plt.figure()
    stack = np.hstack((candidate[:, :, 0], candidate[: ,:, 1]))
    print stack.shape
    plt.imshow(stack, cmap='gray')
    plt.title(score)

def main():
    model_path = '../model/transinet_v1.h5'
    model = keras.models.load_model(model_path)

    dataset_path = sys.argv[1]
    data = np.load(dataset_path)
    class_pred = model.predict_classes(data, verbose=1)
    probs = model.predict(data)
    prob_of_SN = probs.T[0]

    candidate_inds = np.where(prob_of_SN > 0.97)
    for ind in candidate_inds[0]:
        plot_candidate(data[ind, :, :, :], prob_of_SN[ind])
    plt.show()

main()
