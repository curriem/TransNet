import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def gauss_kern(std, kern_len=28):
    """ Returns 2D Gaussian kernel array"""

    gkern_1d = signal.gaussian(kern_len, std=std).reshape(kern_len, 1)
    gkern_2d = np.outer(gkern_1d, gkern_1d)
    return gkern_2d


def circle_kern(radius, kern_len=28):
    center = int((kern_len-1)/2)
    ckern = np.zeros((kern_len, kern_len))
    y, x = np.ogrid[-center: center+1, -center: center+1]
    mask = x**2 + y**2 <= radius**2
    ckern = mask.astype(float)
    return ckern

def square_kern(radius, kern_len=28):
    center = int((kern_len-1)/2)
    skern_1d = np.zeros(kern_len)
    skern_1d[center-radius:center+radius+1] = 1
    skern_2d = np.outer(skern_1d, skern_1d)
    return skern_2d

def add_noise(kern):
    noise = np.random.standard_normal(kern.shape)
    return noise*0.05 + kern


data_stack = []
class_stack = []
for n in range(20000):
    random_std = np.random.random()*5
    random_width = np.random.randint(0, 6)
    random_radius = np.random.randint(0, 10)

    gkern = gauss_kern(random_std)
    skern = square_kern(random_width)
    ckern = circle_kern(random_radius)

    # add noise
    gkern = add_noise(gkern)
    skern = add_noise(skern)
    ckern = add_noise(ckern)

    data_stack.append(gkern)
    class_stack.append(0)
    data_stack.append(skern)
    class_stack.append(1)
    data_stack.append(ckern)
    class_stack.append(2)
data_stack = np.array(data_stack)
np.save('../data/play_data.npy', data_stack)
np.save('../data/play_data_class.npy', class_stack)
