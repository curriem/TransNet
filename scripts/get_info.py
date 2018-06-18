import numpy as np
import sys

idx = int(sys.argv[1])

badpix_dir = '/sets_newbadpix/'

mags = \
np.load('/Users/mcurrie/GitRepos/TransiNet/model/%s/model_mag_test_v2.npy'
        % badpix_dir)

magsort_args = np.argsort(mags)

info = np.load('../data/sets_newbadpix/training_info.npy')

test_inds = np.load('inds_test_rand42_new.npy')

info = info[test_inds]
info = info[magsort_args]

print info[idx]
