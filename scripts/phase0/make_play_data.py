import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob

def gauss_kern(std, kern_len=28):
    """ Returns 2D Gaussian kernel array"""

    gkern_1d = signal.gaussian(kern_len, std=std).reshape(kern_len, 1)
    gkern_2d = np.outer(gkern_1d, gkern_1d)
    return gkern_2d


def circle_kern(radius, kern_len=28):
    center = int((kern_len-1)/2)
    y, x = np.ogrid[-center-1: center+1, -center-1: center+1]
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
    return noise*np.random.random()/5 + kern

for m in range(10):
    data_stack = np.empty((0, 28*28))
    class_stack = np.empty(0, dtype=np.int)
    for n in range(1000):
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

        # flatten
        gkern = gkern.flatten()
        skern = skern.flatten()
        ckern = ckern.flatten()
        

        data_stack = np.append(data_stack, [gkern], axis=0)
        data_stack = np.append(data_stack, [skern], axis=0)
        data_stack = np.append(data_stack, [ckern], axis=0)
        
        class_stack = np.append(class_stack, 0)
        class_stack = np.append(class_stack, 1)
        class_stack = np.append(class_stack, 2)
        print n
    #data_stack = np.array(data_stack)
    #data_stack = data_stack.reshape(60000, 784)
    np.save('../data/play_data%i.npy' % m, data_stack)
    np.save('../data/play_class%i.npy' % m, class_stack)
data_fls = glob.glob('../data/play_data*')
class_fls = glob.glob('../data/play_class*')

data_stack = np.empty((0, 28*28))
class_stack = np.empty(0)
for n in range(len(data_fls)):
    data = np.load(data_fls[n])
    clss = np.load(class_fls[n])
    data_stack = np.concatenate((data_stack, data))
    class_stack = np.concatenate((class_stack, clss))

data_stack = data_stack.astype(np.float32)
class_stack = class_stack.astype(np.int)

np.save('../data/gauss_data.npy', data_stack)
np.save('../data/gauss_labels.npy', class_stack)
